import time
from dataclasses import dataclass

import logging_setup
import requests
from errors import StandFault
from models.registry import EngineKind

from . import ollama, vllm
from .core import CardState, EngineSpec
from .lookup import CARD, card_engines

log = logging_setup.get_logger(__name__)

# ollama let go of the card in 0.7 s once and in over 3 s the next time: the wait polls, never sleeps blind
WAIT_CEILING = 60
POLL_SECONDS = 0.5


class CardNotHanded(StandFault):
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
            if vllm.card_state(spec) in (CardState.AWAKE, CardState.UNKNOWN):
                held.append(Holding(spec, tuple(_served(spec))))
        elif spec.kind is EngineKind.ollama:
            # the same rule as for vLLM: a silence may hold the card, a stopped server does not
            state, seen = ollama.card_reading(spec)
            if state in (CardState.HOLDS, CardState.UNKNOWN):
                held.append(Holding(spec, tuple(m["model"] for m in seen if m["vram_mb"] > 0)))
    return held


# the stack comes up with vLLM awake on the card, and ollama roles would land half on the cpu
def sleep_every_vllm() -> None:
    for spec in card_engines(EngineKind.vllm):
        state = vllm.card_state(spec)
        if state in (CardState.AWAKE, CardState.UNKNOWN):
            # a silent one may be awake, and a refusal to sleep stops the boot rather than hides
            vllm.sleep(spec)
            log.info("card.vllm_asleep", engine=spec.name, was=state)


def _served(spec: EngineSpec) -> list[str]:
    try:
        return vllm.served(spec)
    except Exception:
        return []


# what `model_on_card` read, named beside every reading: a null with no instrument is nowhere to ask
def placement_instrument(spec: EngineSpec) -> str | None:
    if spec.placement not in CARD:
        return "declared placement"
    return {EngineKind.ollama: "ollama /api/ps", EngineKind.vllm: "vllm /is_sleeping"}.get(spec.kind)


# the one instrument: the judge, the run gate and the preflight all read this one
def model_on_card(spec: EngineSpec, model: str) -> bool | None:
    if spec.placement not in CARD:
        return False
    if spec.kind is EngineKind.ollama:
        wanted = ollama.spellings(model)
        seen = [m for m in ollama.residency(spec) if m["model"] in wanted]
        return seen[0]["vram_mb"] >= seen[0]["size_mb"] if seen else None
    if spec.kind is EngineKind.vllm:
        seen = {CardState.AWAKE: True, CardState.ASLEEP: False}.get(vllm.card_state(spec))
        # awake is not enough: the server must serve this very model
        if seen:
            try:
                return model in vllm.served(spec)
            except Exception:
                return None
        return seen
    return None


# half on the processor answers with other kernels; only ollama spills, an asleep vLLM wakes whole
def spilled(spec: EngineSpec, model: str) -> bool:
    return (spec.kind is EngineKind.ollama and spec.placement in CARD
            and model_on_card(spec, model) is False)


# ollama loads a role on its first call, so for it a card nobody else holds is already its own
def holds_for(spec: EngineSpec) -> bool:
    held = on_card()
    if any(h.engine.id != spec.id for h in held):
        return False
    return spec.kind is EngineKind.ollama or any(h.engine.id == spec.id for h in held)


def hand_to(target: EngineSpec, model: str | None = None, allow_spill: bool = False) -> None:
    if target.placement not in CARD:
        # an engine off the card takes nothing from the one on it, and a vLLM there has no sleep
        if target.kind is EngineKind.ollama and model:
            ollama.load_into_memory(model, target)
        return
    _refuse_a_silent_target(target)
    for held in on_card():
        if held.engine.id != target.id:
            _release(held)
    _take(target, model, allow_spill)
    left = [h.engine.name for h in on_card() if h.engine.id != target.id]
    if left:
        raise CardNotHanded(f"{target.name} took the card, and {', '.join(left)} still hold it")
    log.info("card.handed", engine=target.name, model=model)


# asked before anything lets go: a handover to a server that is gone left the card with nobody
def _refuse_a_silent_target(target: EngineSpec) -> None:
    if target.kind is EngineKind.vllm:
        state = vllm.card_state(target)
    elif target.kind is EngineKind.ollama:
        state, _ = ollama.card_reading(target)
    else:
        return
    if state in (CardState.DOWN, CardState.UNKNOWN):
        raise CardNotHanded(f"{target.name} is {state}; the card stays where it is")


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


def _take(target: EngineSpec, model: str | None, allow_spill: bool = False) -> None:
    if target.kind is EngineKind.vllm:
        # a wake on a card not yet free fails and leaves the server asleep; a later one succeeds
        _until(lambda: _woke(target), f"{target.name} did not wake in {WAIT_CEILING}s")
    elif model:
        ollama.load_into_memory(model, target)
        # a model half on the processor answers, with other kernels, in silence
        if spilled(target, model):
            if not allow_spill:
                raise CardNotHanded(f"{model} loaded on {target.name}, but not whole on the card")
            # asked for by `allow_cpu`, and each row's `on_card: false` says so
            log.warning("card.spill_allowed", engine=target.name, model=model)


# a batch of 64 chunks beside a resident 8b dropped ollama's runner with CUDA OOM
def clear_for(spec: EngineSpec, keep: str) -> None:
    if spec.kind is not EngineKind.ollama or spec.placement not in CARD:
        return
    for resident in ollama.residency(spec):
        if resident["model"] not in ollama.spellings(keep):
            ollama.unload(resident["model"], spec)


def _woke(spec: EngineSpec) -> bool:
    if vllm.is_sleeping(spec) is False:
        return True
    try:
        vllm.wake_up(spec)
        return True
    # a server that dies mid-wake answers with a connection error, and the ceiling names it
    except (vllm.WakeFailed, requests.RequestException) as e:
        log.info("card.wake_retry", engine=spec.name, error=str(e))
        return False


def _until(done, why: str) -> None:
    deadline = time.monotonic() + WAIT_CEILING
    while time.monotonic() < deadline:
        if done():
            return
        time.sleep(POLL_SECONDS)
    raise CardNotHanded(why)
