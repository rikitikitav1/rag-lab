import re
import time
from typing import Any

import config
import logging_setup
import requests
from models.registry import EngineKind

from .core import CardState, EngineSpec, NotSupported, Unconfigured, base_url
from .lookup import seeded_ollama

log = logging_setup.get_logger(__name__)

HTTP_TIMEOUT = 60
PULL_TIMEOUT = 3600
# seconds, not a completion's: the card is read every half second while it changes hands
CARD_READ_TIMEOUT = 3


# every call names the server it goes to; the seeded engine answers when nobody named one
def _at(spec: EngineSpec | None) -> str:
    return base_url(spec or seeded_ollama())


def refuse_unless_ollama(spec: EngineSpec | None, doing: str, name: str = "") -> None:
    if spec is not None and spec.kind is not EngineKind.ollama:
        about = f" {name}" if name else ""
        raise NotSupported(f"{doing}{about} is not implemented for {spec.kind} yet")


def _check(response, path) -> Any:
    if not response.ok:
        try:
            error = response.json().get("error", response.text)
        except ValueError:
            error = response.text
        raise RuntimeError(f"Ollama {response.status_code} on {path}: {error}")
    if not response.text:
        return None
    return response.json()


def post(path, payload, spec=None, timeout=HTTP_TIMEOUT):
    return _check(requests.post(f"{_at(spec)}{path}", json=payload, timeout=timeout), path)


def get(path, spec=None) -> dict:
    return _check(requests.get(f"{_at(spec)}{path}", timeout=HTTP_TIMEOUT), path)


# what the server says is loaded right now, or nothing: a probe must not break a run
def loaded_models(spec=None) -> list:
    try:
        return get("/api/ps", spec).get("models") or []
    except Exception as e:
        log.warning("ollama.ps_failed", error=str(e))
        return []


# one row of a run asks, the other 822 read this: the snapshot is written per answer
_WINDOW_SECONDS = 60
_windows: dict[tuple[int | None, str], tuple[float, int | None]] = {}


def context_length(model: str, spec=None) -> int | None:
    key = (spec.id if spec else None, model)
    seen_at, window = _windows.get(key, (0.0, None))
    if time.monotonic() - seen_at < _WINDOW_SECONDS:
        return window
    # the full tag, or llama3.1:8b would read the window of a loaded llama3.1:70b
    wanted = spellings(model)
    window = next((e.get("context_length") for e in loaded_models(spec) if e.get("name") in wanted), None)
    # an unloaded model has no window yet: a cached None hid the real one for a minute after the load
    if window is not None:
        _windows[key] = (time.monotonic(), window)
    return window


def forget_window(model: str, spec=None) -> None:
    _windows.pop((spec.id if spec else None, model), None)


def list_models(spec=None) -> list:
    return [m["name"] for m in list_tagged(spec)]


# the same read of `/api/ps` the card uses, so both treat a silent server alike
def residency(spec=None) -> list[dict]:
    return card_reading(spec)[1]


# a model holds the card while any of it sits in video memory: one rule for the card and a load
def on_card_models(models: list[dict]) -> list[str]:
    return [m["model"] for m in models if m["vram_mb"] > 0]


def card_state(spec=None) -> CardState:
    return card_reading(spec)[0]


# the readings a vLLM gives: a refused connection holds no card, a silence may hold it
def card_reading(spec=None) -> tuple[CardState, list[dict]]:
    try:
        answer = requests.get(f"{_at(spec)}/api/ps", timeout=CARD_READ_TIMEOUT)
        seen = _shaped((_check(answer, "/api/ps") or {}).get("models") or [])
    except requests.Timeout as e:
        log.warning("ollama.ps_unanswered", error=str(e))
        return CardState.UNKNOWN, []
    except (requests.ConnectionError, Unconfigured):
        return CardState.DOWN, []
    except Exception as e:
        log.warning("ollama.ps_unanswered", error=str(e))
        return CardState.UNKNOWN, []
    return (CardState.HOLDS if on_card_models(seen) else CardState.FREE), seen


def _shaped(models: list) -> list[dict]:
    return [
        {
            "model": m.get("name", "?"),
            "size_mb": round((m.get("size") or 0) / 2**20),
            "vram_mb": round((m.get("size_vram") or 0) / 2**20),
        }
        for m in models
        if m.get("size")
    ]


# the registry knows the size before the first byte, and a first pull has no recorded size at all
def registry_size(model: str) -> int | None:
    # a windowed tag lives only here; the registry knows its base, and the bytes are the base's
    name = add_tags([(windowed(model) or (model,))[0]])[0]
    repo, tag = name.rsplit(":", 1)
    if "/" not in repo:
        repo = f"library/{repo}"
    try:
        seen = requests.get(f"https://registry.ollama.ai/v2/{repo}/manifests/{tag}", timeout=10).json()
        return sum(layer["size"] for layer in seen["layers"]) or None
    except Exception as e:
        log.warning("ollama.registry_size_unknown", model=model, error=str(e))
        return None


# a tag that carries its own window: the OpenAI door takes no num_ctx per call, so the window lives in the model
WINDOWED = re.compile(r"^(?P<base>.+:.+)-w(?P<window>\d+)$")


def windowed(model: str) -> tuple[str, int] | None:
    seen = WINDOWED.match(model or "")
    return (seen["base"], int(seen["window"])) if seen else None


def pull_model(model, spec=None):
    derived = windowed(model)
    if derived is None:
        pulled = post("/api/pull", {"model": model, "stream": False}, spec, timeout=PULL_TIMEOUT)
        hold_repetition_penalty(model, spec)
        return pulled
    base, window = derived
    post("/api/pull", {"model": base, "stream": False}, spec, timeout=PULL_TIMEOUT)
    # the base first: the tag made from it carries its penalty along with its own window
    hold_repetition_penalty(base, spec)
    return post(
        "/api/create",
        {"model": model, "from": base, "parameters": {"num_ctx": window}, "stream": False},
        spec,
        timeout=PULL_TIMEOUT,
    )


MEASURED_REPEAT_PENALTY = config.settings.llm.measured_repeat_penalty


def server_version(spec=None) -> str | None:
    try:
        return get("/api/version", spec).get("version")
    except Exception as e:
        log.warning("ollama.version_unknown", error=str(e))
        return None


# `/api/show` lists parameters one to a line, `stop` repeating
def _parameters(seen: dict) -> dict:
    held = {}
    for line in (seen.get("parameters") or "").splitlines():
        name, _, value = line.strip().partition(" ")
        if name:
            held[name] = value.strip().strip('"')
    return held


# a stamp must not die on a silent server: unread is absent, an unmeasured version says so
def repetition_penalty_served(model: str, spec=None):
    try:
        held = _parameters(shown(model, spec)).get("repeat_penalty")
    except Exception as e:
        log.warning("ollama.repetition_penalty_unread", model=model, error=str(e))
        return None
    if held is not None:
        return float(held)
    version = server_version(spec)
    return None if version is None else MEASURED_REPEAT_PENALTY.get(version, "unknown")


# the penalty lives in the model, recreated under its own name; only when it is missing, and the old runner goes
def hold_repetition_penalty(model: str, spec=None) -> bool:
    wanted = config.settings.llm.repetition_penalty
    seen = shown(model, spec)
    if "embedding" in (seen.get("capabilities") or []):
        return False
    held = _parameters(seen).get("repeat_penalty")
    if held is not None and float(held) == wanted:
        return False
    post(
        "/api/create",
        {"model": model, "from": model, "parameters": {"repeat_penalty": wanted}, "stream": False},
        spec,
        timeout=PULL_TIMEOUT,
    )
    # a runner already loaded keeps the parameters it started with until it loads again
    unload(model, spec)
    log.info("ollama.repetition_penalty_held", model=model, was=held, now=wanted)
    return True


def unload(model: str, spec=None) -> None:
    # keep_alive 0 overrides the default; a teardown must not raise on a server already gone
    if spec is not None and spec.kind is not EngineKind.ollama:
        return
    try:
        forget_window(model, spec)
        post("/api/generate", {"model": model, "keep_alive": 0}, spec)
    except Exception as e:
        log.warning("ollama.unload_failed", model=model, error=str(e))


# an empty generate loads the weights and answers nothing
def load_into_memory(model: str, spec=None) -> dict:
    # an embedder answers `/api/generate` with a 400, and an empty embed loads it just the same
    if "embedding" in (shown(model, spec).get("capabilities") or []):
        post("/api/embed", {"model": model, "input": []}, spec)
    else:
        post("/api/generate", {"model": model}, spec)
    log.info("ollama.loaded", model=model)
    forget_window(model, spec)
    return {"model": model, "context_length": context_length(model, spec)}


def delete_model(model, spec=None):
    response = requests.delete(f"{_at(spec)}/api/delete", json={"model": model}, timeout=HTTP_TIMEOUT)
    if response.status_code == 404:
        log.info("ollama.model_already_absent", model=model)
        return None
    return _check(response, "/api/delete")


# what the server says about a model before it is given a role, read rather than assumed
def shown(model: str, spec=None) -> dict:
    return post("/api/show", {"model": model}, spec) or {}


# who serves the window: the configured generator when it is up, else whoever else is
def window_model(configured: str | None, loaded: list[dict] | None = None, spec=None) -> str | None:
    names = [m["model"] for m in (residency(spec) if loaded is None else loaded)]
    if configured in names:
        return configured
    return next((n for n in names if "embed" not in n and "bge" not in n), None)


# what the server knows about weights it already holds: the quant it runs and how big the file is
def artifact_of(model: str, spec=None) -> dict:
    seen = shown(model, spec)
    details = seen.get("details") or {}
    return {
        "quant": details.get("quantization_level"),
        "params": details.get("parameter_size"),
        "family": details.get("family"),
        "size_bytes": _size_of(model, spec),
    }


def _size_of(model: str, spec=None) -> int | None:
    wanted = spellings(model)
    return next((m.get("size") for m in list_tagged(spec) if m.get("name") in wanted), None)


def list_tagged(spec=None) -> list:
    return get("/api/tags", spec)["models"]


def add_tags(models) -> list:
    return [m if ":" in m else f"{m}:latest" for m in models]


# `/api/ps` may add `:latest` to a name the registry keeps bare
def spellings(model: str) -> set[str]:
    return {model, f"{model}:latest"}


def ensure_models(spec=None) -> None:
    for model in set(add_tags(config.settings.llm.pull_models)) - set(add_tags(list_models(spec))):
        pull_model(model, spec)
