import time
from dataclasses import dataclass

import logging_setup
from models.registry import EngineKind

from . import ollama, vllm
from .core import EngineSpec
from .lookup import CARD, card_engines

log = logging_setup.get_logger(__name__)

# ollama let go of the card in 0.7 s once and in over 3 s the next time: the wait polls, never sleeps blind
WAIT_CEILING = 60
POLL_SECONDS = 0.5


class CardNotHanded(RuntimeError):
    pass


@dataclass(frozen=True)
class Holding:
    engine: EngineSpec
    models: tuple[str, ...]


# read from the servers: a table of holders would be a second truth beside them, and they would part
def on_card() -> list[Holding]:
    held = []
    for spec in card_engines():
        if spec.kind is EngineKind.vllm:
            # silence is not a free card: an unanswered server may be awake on it
            if vllm.card_state(spec) in ("awake", "unknown"):
                held.append(Holding(spec, tuple(_served(spec))))
        elif spec.kind is EngineKind.ollama:
            names = tuple(m["model"] for m in ollama.residency(spec) if m["vram_mb"] > 0)
            if names:
                held.append(Holding(spec, names))
    return held


def _served(spec: EngineSpec) -> list[str]:
    try:
        return vllm.served(spec)
    except Exception:
        return []


# where a role's model answered from, by its own engine's instrument: the judge stamps this, now all do
def model_on_card(spec: EngineSpec, model: str) -> bool | None:
    if spec.placement not in CARD:
        return False
    if spec.kind is EngineKind.ollama:
        wanted = {model, f"{model}:latest"}
        seen = [m for m in ollama.residency(spec) if m["model"] in wanted]
        return seen[0]["vram_mb"] >= seen[0]["size_mb"] if seen else None
    if spec.kind is EngineKind.vllm:
        return {"awake": True, "asleep": False}.get(vllm.card_state(spec))
    return None


# ollama loads a role on its first call, so for it a card nobody else holds is already its own
def holds_for(spec: EngineSpec) -> bool:
    held = on_card()
    if any(h.engine.id != spec.id for h in held):
        return False
    return spec.kind is EngineKind.ollama or any(h.engine.id == spec.id for h in held)


def hand_to(target: EngineSpec, model: str | None = None) -> None:
    if target.placement not in CARD:
        # an engine off the card takes nothing from the one on it, and a vLLM there has no sleep
        if target.kind is EngineKind.ollama and model:
            ollama.load_into_memory(model, target)
        return
    for held in on_card():
        if held.engine.id != target.id:
            _release(held)
    _take(target, model)
    left = [h.engine.name for h in on_card() if h.engine.id != target.id]
    if left:
        raise CardNotHanded(f"{target.name} took the card, and {', '.join(left)} still hold it")
    log.info("card.handed", engine=target.name, model=model)


def _release(held: Holding) -> None:
    if held.engine.kind is EngineKind.vllm:
        vllm.sleep(held.engine)
    else:
        for name in held.models:
            ollama.unload(name, held.engine)
    _until(
        lambda: all(h.engine.id != held.engine.id for h in on_card()),
        f"{held.engine.name} did not let go of the card in {WAIT_CEILING}s",
    )


def _take(target: EngineSpec, model: str | None) -> None:
    if target.kind is EngineKind.vllm:
        # a wake on a card not yet free fails and leaves the server asleep; a later one succeeds
        _until(lambda: _woke(target), f"{target.name} did not wake in {WAIT_CEILING}s")
    elif model:
        ollama.load_into_memory(model, target)


def _woke(spec: EngineSpec) -> bool:
    if vllm.is_sleeping(spec) is False:
        return True
    try:
        vllm.wake_up(spec)
        return True
    except vllm.WakeFailed as e:
        log.info("card.wake_retry", engine=spec.name, error=str(e))
        return False


def _until(done, why: str) -> None:
    deadline = time.monotonic() + WAIT_CEILING
    while time.monotonic() < deadline:
        if done():
            return
        time.sleep(POLL_SECONDS)
    raise CardNotHanded(why)
