import contextlib
import os
from dataclasses import dataclass
from typing import Any

import config
import engines
import logging_setup
from engines import vllm as vllm_engine
from engines.lookup import Resolved
from errors import StandFault
from models.registry import EngineKind
from openai import APIStatusError, OpenAIError

# where the seeded engine answers; every other engine says so through its own `env_prefix`
LLM_BASE = os.getenv("OLLAMA_BASE_URL") or config.settings.llm.base_url

log = logging_setup.get_logger(__name__)

# set by the worker alone: a job that calls a second role on the card takes it per call, the API never
_before_call = None


def take_the_card_before_calls(hook) -> None:
    global _before_call
    _before_call = hook


# the hook may hand back what ends the call's hold on its engine, called once the answer is in
@contextlib.contextmanager
def _card_for(spec, name: str):
    ended = _before_call(spec, name) if _before_call is not None else None
    try:
        yield
    finally:
        if ended is not None:
            ended()


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ChatTurn:
    text: str | None
    tool_calls: list
    message: Any
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None


def resolve(role: str) -> Resolved:
    return engines.spec_of_role(role)


def resolve_name(role: str) -> str:
    return resolve(role).name


# a bare name was unambiguous while one engine held every model; two engines make it a question
def resolve_for(role: str, model: str | None = None) -> Resolved:
    if model is None:
        return resolve(role)
    mine = resolve(role)
    try:
        found = engines.find_model(model)
    except engines.Ambiguous:
        # the override names a model, not an engine: the role's own engine answers if it has it
        found = engines.find_model(model, mine.engine.id)
        if found is None:
            raise
    # a name the registry never saw still runs, on the engine the role would have used
    return found or Resolved(model, mine.engine)


# the upstream body can carry a fragment of the key we kept out of the database on purpose
def _without_the_body(e: Exception) -> str:
    if isinstance(e, APIStatusError):
        return f"http {e.status_code}"
    return type(e).__name__


# one contract for a failed completion: the same log event and error text, written twice
def _complete(spec, name: str, messages, params):
    with _card_for(spec, name):
        try:
            return engines.client_for(spec).chat.completions.create(
                model=name, messages=messages, **params
            )
        except OpenAIError as e:
            said = _without_the_body(e)
            log.error("llm.chat_failed", model=name, engine=spec.name, error=said)
            raise RuntimeError(f"LLM chat failed ({name} on {spec.name}): {said}") from e


def ask(system, user, role="generation", schema=None, model=None) -> Completion:
    picked = resolve_for(role, model)
    name = picked.name
    resp = _complete(
        picked.engine,
        name,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        _params(role, schema, picked.engine),
    )

    usage = resp.usage
    log.info(
        "llm.chat",
        model=name,
        engine=picked.engine.name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    return Completion(
        text=resp.choices[0].message.content,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


def chat(messages, tools=None, role="generation", model=None) -> ChatTurn:
    picked = resolve_for(role, model)
    name = picked.name
    params = _params(role, None, picked.engine)
    if tools:
        params["tools"] = tools
    resp = _complete(picked.engine, name, messages, params)

    choice = resp.choices[0]
    message = choice.message
    usage = resp.usage
    log.info(
        "llm.chat_tools",
        model=name,
        engine=picked.engine.name,
        tool_calls=len(message.tool_calls or []),
        finish_reason=choice.finish_reason,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    return ChatTurn(
        text=message.content,
        tool_calls=message.tool_calls or [],
        message=message,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        finish_reason=choice.finish_reason,
    )


# the same dict the call is made with, so a stamp cannot drift from what was sent
def sampler_of(role, spec=None) -> dict:
    return sampler(role, spec).sent


# both halves in one read: what went out, and what the role asked for and the engine refused
def sampler(role, spec=None) -> engines.Sampler:
    opts = config.settings.llm.roles[role].options
    # at temperature 0 the sampler does not roll, but a batching server needs the run to say
    wanted = {k: opts[k] for k in engines.SAMPLER_KEYS if k in opts}
    return engines.translate(spec or resolve(role).engine, wanted)


# `response_format` and `tools` bypass `translate`: an engine that cannot do them refuses loudly
def _params(role, schema, spec) -> dict:
    params = sampler(role, spec).sent
    if schema:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": schema},
        }
    return params


# the cross-encoder is a role like the others: its engine answers, and the card is taken for it
def score_pairs(pairs: list, role="reranking") -> list[float]:
    if not pairs:
        # the server refuses an empty list with a 400, and nothing is there to score
        return []
    picked = resolve(role)
    name, spec = picked.name, picked.engine
    if spec.kind is not EngineKind.vllm:
        raise RuntimeError(f"{name} on {spec.name}: only a vLLM pooling server scores pairs")
    with _card_for(spec, name):
        try:
            scores = vllm_engine.score(spec, name, pairs)
        except StandFault:
            raise
        except Exception as e:
            said = _without_the_body(e)
            log.error("llm.rerank_failed", model=name, engine=spec.name, error=said)
            raise RuntimeError(f"LLM rerank failed ({name} on {spec.name}): {said}") from e
    log.info("llm.rerank", model=name, engine=spec.name, count=len(pairs))
    return scores


# what wrote a vector: one model name on two engines writes two geometries
def embedder_label(role="embedding") -> str:
    picked = resolve(role)
    return engines.label(picked.name, picked.engine.name)


def embed(prompt, role="embedding"):
    return request_embeddings_batch([prompt], role)[0]


def request_embeddings_batch(texts, role="embedding"):
    return _embeddings(resolve(role), texts)


# the label and the vectors from one resolution: read apart, a role seated between them mislabels
def embed_labelled(texts, role="embedding") -> tuple[str, list]:
    picked = resolve(role)
    return engines.label(picked.name, picked.engine.name), _embeddings(picked, texts)


def _embeddings(picked, texts) -> list:
    name = picked.name
    with _card_for(picked.engine, name):
        try:
            resp = engines.client_for(picked.engine).embeddings.create(model=name, input=texts)
        except OpenAIError as e:
            said = _without_the_body(e)
            log.error("llm.embed_failed", model=name, engine=picked.engine.name, error=said)
            raise RuntimeError(f"LLM embed failed ({name} on {picked.engine.name}): {said}") from e
    log.info("llm.embed", model=name, engine=picked.engine.name, count=len(texts))
    return [d.embedding for d in resp.data]
