import os
import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

import config
import samplers
from models.registry import EngineKind, Placement
from openai import OpenAI
from redaction import redact

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

SAMPLER_KEYS = samplers.KEYS

# full in every row today, so nothing is dropped: a real limit belongs to an engine, not a kind
ACCEPTS = {
    EngineKind.ollama: frozenset(SAMPLER_KEYS),
    EngineKind.vllm: frozenset(SAMPLER_KEYS),
    EngineKind.openai_compatible: frozenset(SAMPLER_KEYS),
}

# the seeded engine, whose address predates the table and still comes from the old variable
SEEDED_PREFIX = "OLLAMA"


class Unnamed(ValueError):
    pass


class Ambiguous(ValueError):
    pass


# what a server says about the card: a refused connection is down, a silence is unknown
class CardState(StrEnum):
    AWAKE = "awake"
    ASLEEP = "asleep"
    HOLDS = "holds"
    FREE = "free"
    DOWN = "down"
    UNKNOWN = "unknown"


# gone, or silent past its timeout: nothing is handed to it and no role is seated on it
SILENT = frozenset({CardState.DOWN, CardState.UNKNOWN})


class Unconfigured(RuntimeError):
    pass


# the management half speaks ollama alone; every door translates this into its own shape
class NotSupported(RuntimeError):
    pass


# a detached snapshot: the row is read once and the session closes before the call goes out
@dataclass(frozen=True)
class EngineSpec:
    id: int
    name: str
    kind: EngineKind
    env_prefix: str
    placement: Placement


@dataclass(frozen=True)
class Sampler:
    sent: dict
    # what the role asked for and the engine will not carry, so a stamp cannot claim it applied
    dropped: dict


def translate(spec: EngineSpec, wanted: dict) -> Sampler:
    accepts = ACCEPTS[spec.kind]
    return Sampler(
        sent={k: v for k, v in wanted.items() if k in accepts},
        dropped={k: v for k, v in wanted.items() if k not in accepts},
    )


# the row carries a prefix, not an address: a record you can read a key out of leaks through reports
def base_url(spec: EngineSpec) -> str:
    seen = os.getenv(f"{spec.env_prefix}_BASE_URL")
    if not seen and spec.env_prefix == SEEDED_PREFIX:
        # only the seeded row falls back: a typo in a second prefix must not address the first
        seen = config.settings.llm.base_url
    if not seen:
        raise Unconfigured(f"engine {spec.name}: address is not configured")
    # a broker's page gives the address with /v1, and the client adds its own
    seen = seen.rstrip("/")
    seen = seen[: -len("/v1")] if seen.endswith("/v1") else seen
    return _refuse_unusable(spec, seen)


# a broker, not a server of ours: the one kind that bills, caches and keeps no residency
def is_cloud(kind: EngineKind) -> bool:
    return kind is EngineKind.openai_compatible


# a scheme we do not speak, or a key smuggled in the address, is a refusal before the first call
def _refuse_unusable(spec: EngineSpec, address: str) -> str:
    seen = urlsplit(address)
    if seen.scheme not in ("http", "https") or not seen.hostname:
        raise Unconfigured(f"engine {spec.name}: address is not an http url")
    if seen.username or seen.password:
        raise Unconfigured(f"engine {spec.name}: address carries credentials")
    if spec.placement is Placement.remote and seen.scheme != "https":
        raise Unconfigured(f"engine {spec.name}: a remote engine takes https, or its key travels in the clear")
    if redact(address) != address:
        raise Unconfigured(f"engine {spec.name}: address carries a key; put it in {spec.env_prefix}_API_KEY")
    return address


# `model@engine`: the embedder's mark on every vector, the roles' drift, the migration's backfill
def label(model: str, engine: str) -> str:
    return f"{model}@{engine}"


# a paid engine refuses here rather than sending a placeholder and reading the server's 401
def api_key(spec: EngineSpec) -> str:
    seen = os.getenv(f"{spec.env_prefix}_API_KEY")
    if seen:
        return seen
    if is_cloud(spec.kind):
        raise Unconfigured(f"engine {spec.name}: key is not configured")
    # deliberately the kind's name: a local server ignores it, and changing it would move the wire
    return spec.kind.value


# host and port only: `netloc` carries userinfo, and this lands on every judged row
def address_of(spec: EngineSpec) -> str:
    seen = urlsplit(base_url(spec))
    # a base without an explicit port stamped every row with the string `host:None`
    return f"{seen.hostname}:{seen.port}" if seen.port else seen.hostname


_clients: dict[int, OpenAI] = {}

# a broker drops a call now and then and answers the next one; every 5xx past the retries stops the run
CLOUD_RETRIES = 4


def _retries(spec: EngineSpec) -> int:
    return CLOUD_RETRIES if is_cloud(spec.kind) else 1


def client_for(spec: EngineSpec) -> OpenAI:
    got = _clients.get(spec.id)
    if got is None:
        got = _clients[spec.id] = OpenAI(
            base_url=f"{base_url(spec)}/v1",
            api_key=api_key(spec),
            timeout=LLM_TIMEOUT,
            max_retries=_retries(spec),
        )
    return got


# an edited engine keeps answering from a long-lived worker until its client is dropped
def forget_clients(engine_id: int | None = None) -> None:
    if engine_id is None:
        _clients.clear()
    else:
        _clients.pop(engine_id, None)


# what the engine adds that the role never asked for, read from the server rather than from compose
def added_by(spec: EngineSpec, model: str) -> dict:
    if spec.kind is EngineKind.ollama:
        from . import ollama

        return _named({"num_ctx": ollama.context_length(model, spec)})
    # a paid engine's host is not asked vLLM's routes, and its key goes to nothing but its calls
    if spec.kind is not EngineKind.vllm:
        return {}
    from . import vllm

    started = vllm.started_at(spec)
    return _named({
        "max_model_len": vllm.max_model_len(spec, model),
        "engine_version": _asked(spec, "/version", "version"),
        # the flag every vLLM noise floor rests on, and nothing in the record said whether it was on
        "batch_invariant": _vllm_env(spec).get("VLLM_BATCH_INVARIANT"),
        "started_at": started,
        "tool_calls_probed": vllm.known_probe(spec, model, started),
        **_weights_as_served(spec),
    })


_SERVED_AS = re.compile(r"\b(dtype|quantization|kv_cache_dtype)=([^,)\s]+)")
_GRAMMAR = re.compile(r"StructuredOutputsConfig\(([^)]*)\)")
_GRAMMAR_AS = re.compile(r"\b(backend|disable_any_whitespace)=([^,)\s]+)")


# what the server loaded, not what the weights declare: `--dtype auto` and an AWQ checkpoint settle it at load
def _weights_as_served(spec: EngineSpec) -> dict:
    said = str(_asked(spec, "/server_info", "vllm_config") or "")
    seen = {}
    for key, value in _SERVED_AS.findall(said):
        if value != "None":
            seen.setdefault(key, value.strip("'\"").removeprefix("torch."))
    # the rules a judge's JSON is decoded by: free whitespace let one reply write tabs until its limit
    if grammar := _GRAMMAR.search(said):
        seen |= {f"json_{key}": value.strip("'\"") for key, value in _GRAMMAR_AS.findall(grammar.group(1))}
    return seen


# `/server_info` exists only under VLLM_SERVER_DEV_MODE, and its absence is silence, not a false
def _vllm_env(spec: EngineSpec) -> dict:
    seen = _asked(spec, "/server_info", "vllm_env")
    return seen if isinstance(seen, dict) else {}


# absent, not null: a key with no value says the server was asked and answered nothing
def _named(seen: dict) -> dict:
    return {k: v for k, v in seen.items() if v is not None}


def _asked(spec: EngineSpec, path: str, key: str):
    try:
        import requests

        return requests.get(f"{base_url(spec)}{path}", headers=_auth(spec), timeout=5).json().get(key)
    except Exception:
        return None


# the one header every engine call carries; a missing key raises, as the key itself does
def bearer(spec: EngineSpec) -> dict:
    return {"Authorization": f"Bearer {api_key(spec)}"}


# a server started with `--api-key` answers nothing without it, and a missing key sends none
def _auth(spec: EngineSpec) -> dict:
    try:
        return bearer(spec)
    except Unconfigured:
        return {}


# `/v1/models` with the key, raising as requests does, so a caller tells silence from a refusal
def models_listing(spec: EngineSpec, timeout: float) -> list[dict]:
    import requests

    seen = requests.get(f"{base_url(spec)}/v1/models", headers=bearer(spec), timeout=timeout)
    seen.raise_for_status()
    return seen.json()["data"]


# the ids a server lists, asked in seconds; None when it does not answer, and a missing key raises
def served_models(spec: EngineSpec, timeout: float = 3) -> list[str] | None:
    try:
        return [m.get("id") for m in models_listing(spec, timeout)]
    except Unconfigured:
        raise
    except Exception:
        return None
