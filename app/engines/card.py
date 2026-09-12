import time
from dataclasses import dataclass

import logging_setup
import requests
from errors import StandFault
from models.registry import EngineKind

from . import vllm
from .core import SILENT, CardState, EngineSpec
from .drivers import driver
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
        models = driver(spec.kind).holding(spec)
        if models is not None:
            held.append(Holding(spec, models))
    return held


# the stack comes up with vLLM awake on the card, and ollama roles would land half on the cpu
def sleep_every_vllm() -> None:
    for spec in card_engines(EngineKind.vllm):
        state = vllm.card_state(spec)
        if state in (CardState.AWAKE, CardState.UNKNOWN):
            # a silent one may be awake, and a refusal to sleep stops the boot rather than hides
            try:
                vllm.sleep(spec)
            except requests.Timeout as e:
                raise CardNotHanded(
                    f"{spec.name} did not go to sleep in {vllm.WAKE_TIMEOUT}s; the boot stops here"
                ) from e
            log.info("card.vllm_asleep", engine=spec.name, was=state)


# what `model_on_card` read, named beside every reading: a null with no instrument is nowhere to ask
def placement_instrument(spec: EngineSpec) -> str | None:
    if spec.placement not in CARD:
        return "declared placement"
    return driver(spec.kind).instrument


# the one instrument: the judge, the run gate and the preflight all read this one
def model_on_card(spec: EngineSpec, model: str) -> bool | None:
    if spec.placement not in CARD:
        return False
    return driver(spec.kind).on_card(spec, model)


# half on the processor answers with other kernels; only ollama spills, an asleep vLLM wakes whole
def spilled(spec: EngineSpec, model: str) -> bool:
    return _can_spill(spec) and model_on_card(spec, model) is False


# the same rule over a reading already taken, so one role is not asked twice
def spilled_reading(spec: EngineSpec, on: bool | None) -> bool:
    return _can_spill(spec) and on is False


def _can_spill(spec: EngineSpec) -> bool:
    return driver(spec.kind).spills and spec.placement in CARD


def holds_for(spec: EngineSpec) -> bool:
    held = on_card()
    if any(h.engine.id != spec.id for h in held):
        return False
    return driver(spec.kind).owns_a_free_card or any(h.engine.id == spec.id for h in held)


def hand_to(target: EngineSpec, model: str | None = None, allow_spill: bool = False) -> None:
    if target.placement not in CARD:
        # an engine off the card takes nothing from the one on it, and a vLLM there has no sleep
        driver(target.kind).load_off_card(target, model)
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
    state = driver(target.kind).state(target)
    if state in SILENT:
        raise CardNotHanded(f"{target.name} is {state}; the card stays where it is")


def _release(held: Holding) -> None:
    driver(held.engine.kind).let_go(held.engine, held.models)
    _until(
        lambda: all(h.engine.id != held.engine.id for h in on_card()),
        f"{held.engine.name} did not let go of the card in {WAIT_CEILING}s",
    )


def _take(target: EngineSpec, model: str | None, allow_spill: bool = False) -> None:
    taking = driver(target.kind)
    _until(lambda: taking.take_once(target, model),
           f"{target.name} did not {taking.takes_the_card} in {WAIT_CEILING}s")
    # a model half on the processor answers, with other kernels, in silence
    if model and spilled(target, model):
        if not allow_spill:
            raise CardNotHanded(f"{model} loaded on {target.name}, but not whole on the card")
        # asked for by `allow_cpu`, and each row's `on_card: false` says so
        log.warning("card.spill_allowed", engine=target.name, model=model)


# a role's model out of memory after its job; the unload logs its own failures, so no job dies here
def release_model(spec: EngineSpec, model: str) -> None:
    driver(spec.kind).unload(spec, model)


def clear_for(spec: EngineSpec, keep: str) -> None:
    if spec.placement in CARD:
        driver(spec.kind).make_room(spec, keep)


def _until(done, why: str) -> None:
    deadline = time.monotonic() + WAIT_CEILING
    while time.monotonic() < deadline:
        if done():
            return
        time.sleep(POLL_SECONDS)
    raise CardNotHanded(why)
