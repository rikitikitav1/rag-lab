import os
from dataclasses import dataclass
from typing import Any

import config
import engines
import logging_setup
from engines.lookup import Resolved
from openai import APIStatusError, OpenAIError

# where the seeded engine answers; every other engine says so through its own `env_prefix`
LLM_BASE = os.getenv("OLLAMA_BASE_URL") or config.settings.llm.base_url

log = logging_setup.get_logger(__name__)


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


def embed(prompt, role="embedding"):
    return request_embeddings_batch([prompt], role)[0]


def request_embeddings_batch(texts, role="embedding"):
    picked = resolve(role)
    name = picked.name
    try:
        resp = engines.client_for(picked.engine).embeddings.create(model=name, input=texts)
    except OpenAIError as e:
        said = _without_the_body(e)
        log.error("llm.embed_failed", model=name, engine=picked.engine.name, error=said)
        raise RuntimeError(f"LLM embed failed ({name} on {picked.engine.name}): {said}") from e
    log.info("llm.embed", model=name, engine=picked.engine.name, count=len(texts))
    return [d.embedding for d in resp.data]
