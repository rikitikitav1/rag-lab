import contextlib
import contextvars
import threading
from dataclasses import dataclass
from typing import Any

import config
import engines
import logging_setup
import token_fields
from engines import answer_parsers
from engines import vllm as vllm_engine
from engines.lookup import Resolved
from errors import StandFault
from models.registry import EngineKind
from openai import APIStatusError, OpenAIError

# where the seeded engine answers; every other engine says so through its own `env_prefix`

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
    parsed: answer_parsers.Parsed | None = None
    parser: str = answer_parsers.NO_PARSER
    finish_reason: str | None = None


@dataclass
class ChatTurn:
    text: str | None
    tool_calls: list
    message: Any
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None
    parsed: answer_parsers.Parsed | None = None


# a broker without usage fails every later row the same way, so the run stops on the first one
class NoUsage(StandFault):
    pass


# the client already asked once more; past that, a quota or a rate limit refuses every row alike
class BrokerRefused(StandFault):
    pass


# a revoked or wrong key refuses every later row alike, and a seat on it is refused at acceptance
class KeyRefused(BrokerRefused):
    pass


# after the client's own retry a 5xx is the server saying it is broken, and the next row meets the same
class ServerFailed(StandFault):
    pass


# one reading of a failed call for chat and embeddings: which failures stop the run and which fail one row
def _failed(e: OpenAIError, spec, name: str, what: str) -> Exception:
    said = _without_the_body(e)
    status = e.status_code if isinstance(e, APIStatusError) else None
    if status in (401, 403):
        log.error("llm.key_refused", model=name, engine=spec.name, status=status)
        return KeyRefused(f"{spec.name} refused the key for {name}: {said}")
    if status in (402, 429):
        log.error("llm.broker_refused", model=name, engine=spec.name, status=status)
        return BrokerRefused(f"{spec.name} refused {name}: {said}, a quota or a rate limit")
    # a local 500 can be one reply the server could not build, as ollama's on a broken tool call
    if status is not None and status >= 500 and engines.is_cloud(spec.kind):
        log.error("llm.server_failed", model=name, engine=spec.name, status=status)
        return ServerFailed(f"{spec.name} failed {name}: {said}, the server says it is broken")
    log.error(f"llm.{what}_failed", model=name, engine=spec.name, error=said)
    return RuntimeError(f"LLM {what} failed ({name} on {spec.name}): {said}")


# the stand counts tokens on every call, so a broker that sends no usage is named, not read as zero
def _usage(resp, engine):
    if resp.usage is None:
        raise NoUsage(f"{engine.name} returned no token usage")
    return resp.usage


# what a job or a row spent, per role and per engine and model; pool threads add to it under the lock
class Tally:
    def __init__(self):
        self._lock = threading.Lock()
        self._seen: dict[tuple[str, str, str], dict[str, int]] = {}

    # the longest input against the window, and the calls the output limit cut: a sum hides both
    def add(self, role: str, engine: str, model: str, prompt: int | None, completion: int | None,
            cut: bool = False) -> None:
        with self._lock:
            got = self._seen.setdefault((role, engine, model), dict.fromkeys(token_fields.FIELDS, 0))
            got["prompt"] += prompt or 0
            got["completion"] += completion or 0
            got["calls"] += 1
            got["uncounted"] += prompt is None
            got["max_prompt"] = max(got["max_prompt"], prompt or 0)
            got["cut_by_length"] += cut

    def record(self) -> dict | None:
        with self._lock:
            out: dict[str, list[dict]] = {}
            for (role, engine, model), counts in sorted(self._seen.items()):
                out.setdefault(role, []).append({
                    "engine": engine, "model": model,
                    **{k: v for k, v in counts.items() if v or k not in token_fields.OPTIONAL},
                })
            return out or None


# a context, not a global: the worker runs two lanes as two threads of one process
_tallies: contextvars.ContextVar[tuple] = contextvars.ContextVar("llm_tallies", default=())

# a broker answers a repeated body from its cache, so a cloud call carries its job in `user`: a second pass is another job
_cache_key: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_cache_key", default=None)
CACHE_KEY = "user=job"


@contextlib.contextmanager
def cache_keyed(key: str | None):
    token = _cache_key.set(key)
    try:
        yield
    finally:
        _cache_key.reset(token)


# only a broker keeps such a cache, and the model never sees the field
def cache_key_of(spec) -> str | None:
    return CACHE_KEY if engines.is_cloud(spec.kind) else None


def _keyed(spec) -> dict:
    key = _cache_key.get()
    return {"user": key} if key and cache_key_of(spec) else {}


# scopes nest: a guest row counts into its own tally and into the job's at once
@contextlib.contextmanager
def accounting(tally: Tally | None = None):
    tally = tally or Tally()
    token = _tallies.set((*_tallies.get(), tally))
    try:
        yield tally
    finally:
        _tallies.reset(token)


# a pool thread starts with an empty context, so the task takes the caller's tallies and key along
def carried(fn):
    tallies, key = _tallies.get(), _cache_key.get()

    def run(*args, **kwargs):
        counted, keyed = _tallies.set(tallies), _cache_key.set(key)
        try:
            return fn(*args, **kwargs)
        finally:
            _cache_key.reset(keyed)
            _tallies.reset(counted)

    return run


def _count(role, engine, model: str, prompt: int | None, completion: int | None,
           finish_reason: str | None = None) -> None:
    for tally in _tallies.get():
        tally.add(str(getattr(role, "value", role)), engine.name, model, prompt, completion,
                  cut=token_fields.cut(finish_reason))


# a call by engine and name, for a model no role holds yet: the probe before a seat
def complete_on(spec, name: str, messages, params, role):
    resp = _complete(spec, name, messages, params)
    usage = _usage(resp, spec)
    _count(role, spec, name, usage.prompt_tokens, usage.completion_tokens,
           getattr(resp.choices[0], "finish_reason", None) if resp.choices else None)
    log.info("llm.chat", role=str(getattr(role, "value", role)), model=name, engine=spec.name,
             prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)
    return resp


# one cut for every answer, after the call and before the judge, the guest or the agent reads it
def _parsed(picked, message, finish_reason=None) -> answer_parsers.Parsed:
    parsed = answer_parsers.parse(picked.parser, message.content,
                                  getattr(message, "reasoning_content", None), finish_reason)
    if parsed.leftover_markers:
        log.warning("llm.leftover_markers", model=picked.name, engine=picked.engine.name,
                    parser=picked.parser, markers=list(parsed.leftover_markers))
    return parsed


def resolve(role: str) -> Resolved:
    return engines.spec_of_role(role)


def resolve_name(role: str) -> str:
    return resolve(role).name


# a bare name was unambiguous while one engine held every model; two engines make it a question
def resolve_for(role: str, model: str | None = None) -> Resolved:
    if model is None:
        return resolve(role)
    mine = resolve(role)
    # the override names a model, not an engine: the role's own engine answers if it has it
    found = engines.find_model_on(model, lambda: mine.engine.id)
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
                model=name, messages=messages, **params, **_keyed(spec)
            )
        except OpenAIError as e:
            raise _failed(e, spec, name, "chat") from e


def ask(system, user, role="generation", schema=None, model=None) -> Completion:
    picked = resolve_for(role, model)
    name = picked.name
    # no system at all is not an empty one: a template drops its default only for a system it was given
    messages = ([] if system is None else [{"role": "system", "content": system}]) + [{"role": "user", "content": user}]
    resp = _complete(picked.engine, name, messages, _params(role, schema, picked))

    usage = _usage(resp, picked.engine)
    _count(role, picked.engine, name, usage.prompt_tokens, usage.completion_tokens,
           getattr(resp.choices[0], "finish_reason", None))
    log.info(
        "llm.chat",
        role=role,
        model=name,
        engine=picked.engine.name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    parsed = _parsed(picked, resp.choices[0].message, getattr(resp.choices[0], "finish_reason", None))
    return Completion(
        text=parsed.text,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        parsed=parsed,
        parser=answer_parsers.label(picked.parser),
        finish_reason=getattr(resp.choices[0], "finish_reason", None),
    )


def chat(messages, tools=None, role="generation", model=None) -> ChatTurn:
    picked = resolve_for(role, model)
    name = picked.name
    params = _params(role, None, picked)
    if tools:
        params["tools"] = tools
    resp = _complete(picked.engine, name, messages, params)

    choice = resp.choices[0]
    message = choice.message
    usage = _usage(resp, picked.engine)
    _count(role, picked.engine, name, usage.prompt_tokens, usage.completion_tokens, choice.finish_reason)
    log.info(
        "llm.chat_tools",
        role=role,
        model=name,
        engine=picked.engine.name,
        tool_calls=len(message.tool_calls or []),
        finish_reason=choice.finish_reason,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    parsed = _parsed(picked, message, choice.finish_reason)
    return ChatTurn(
        text=parsed.text,
        tool_calls=message.tool_calls or [],
        message=message,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        finish_reason=choice.finish_reason,
        parsed=parsed,
    )


# the same dict the call is made with, so a stamp cannot drift from what was sent
def sampler_of(role, picked=None) -> dict:
    return sampler(role, picked).sent


# both halves in one read: what went out, and what the role and the model asked for and the engine refused
def sampler(role, picked=None) -> engines.Sampler:
    picked = picked or resolve(role)
    opts = {**config.settings.llm.roles[role].options, **(getattr(picked, "options", None) or {})}
    # at temperature 0 the sampler does not roll, but a batching server needs the run to say
    wanted = {k: opts[k] for k in engines.SAMPLER_KEYS if k in opts}
    return engines.translate(picked.engine, wanted)


# `response_format` and `tools` bypass `translate`: an engine that cannot do them refuses loudly
def _params(role, schema, picked) -> dict:
    params = sampler(role, picked).sent
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


# unlabelled, for vectors nobody stores, as the guest's; a writer of vectors takes `embed_labelled`
def request_embeddings_batch(texts, role="embedding"):
    return _embeddings(resolve(role), texts, role)


# the label and the vectors from one resolution: read apart, a role seated between them mislabels
def embed_labelled(texts, role="embedding") -> tuple[str, list]:
    picked = resolve(role)
    return engines.label(picked.name, picked.engine.name), _embeddings(picked, texts, role)


# one text: the vector a search asks with and the label the rows it meets must carry
def embed_with_label(text, role="embedding") -> tuple[str, list]:
    label, vectors = embed_labelled([text], role)
    return label, vectors[0]


def _embeddings(picked, texts, role="embedding") -> list:
    name = picked.name
    with _card_for(picked.engine, name):
        try:
            resp = engines.client_for(picked.engine).embeddings.create(model=name, input=texts)
        except OpenAIError as e:
            raise _failed(e, picked.engine, name, "embed") from e
    # on a cloud the tokens are the quota, so a reply without them stops the run; locally it is a gap
    usage = _usage(resp, picked.engine) if engines.is_cloud(picked.engine.kind) else getattr(resp, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    _count(role, picked.engine, name, prompt, 0)
    log.info("llm.embed", role=role, model=name, engine=picked.engine.name, count=len(texts), prompt_tokens=prompt)
    return [d.embedding for d in resp.data]
