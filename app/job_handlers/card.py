import threading

import engines
import llm
from engines import card, ollama, vllm
from models.registry import EngineKind, Role
from use_cases import model_acceptance

from .base import Final, register

_taking = threading.Lock()


# the one road a worker takes to the card: a preamble, a second role's call and the API's ask
def take(spec, model: str | None = None, allow_spill: bool = False) -> None:
    with _taking:
        try:
            if spec.placement in engines.CARD and card.holds_for(spec) and not _to_load(spec, model):
                return
            card.hand_to(spec, model, allow_spill=allow_spill)
        except card.CardNotHanded:
            raise
        except Exception as e:
            raise card.CardNotHanded(f"the card did not reach {spec.name}: {e}") from e
        # after the handover, so a failed probe does not read as a lost card
        _probe_the_woken_generator(spec)


# ollama holding the card may still lack the model a load asked for; a vLLM serves its one model
def _to_load(spec, model: str | None) -> bool:
    if spec.kind is not EngineKind.ollama or not model:
        return False
    # all of it on the processor is not loaded for the card: the handover frees it and loads again
    resident = {m["model"] for m in ollama.residency(spec) if m["vram_mb"] > 0}
    return not resident & ollama.spellings(model)


# asked from the API, which has nothing to wait with; a job takes the card itself, in its own turn
@register("hand_card")
def hand_card(options: dict) -> None:
    target = engines.spec_of_id(options["engine_id"])
    if target is None:
        raise ValueError(f"engine {options['engine_id']} is not registered")
    take(target, options.get("model"))
    if options.get("seat"):
        role = Role(options["seat"])
        try:
            # awake now, so the door's own gate asks the server and records what it said
            model_acceptance.refuse_unfit_model(role, options["model"], target.id)
            model_acceptance.seat(role, target.id, options["model"], over=options.get("seat_over"))
        except ValueError as e:
            raise Final(str(e)) from e


# the gate cannot ask an asleep server, so the worker asks once the card is its; the stamp reads it
def _probe_the_woken_generator(spec) -> None:
    if spec.kind is not EngineKind.vllm:
        return
    try:
        picked = llm.resolve("generation")
    except engines.Unnamed:
        return
    if picked.engine.id == spec.id:
        vllm.tool_calls_probed(spec, picked.name)


# a second role on another card engine is allowed at its cost: the same road, per call
def take_for_call(spec, _model: str) -> None:
    if spec.placement in engines.CARD:
        # no model: ollama loads on the call itself, and an embedder cannot be loaded by a generate
        take(spec)


def clear_the_engine_for(role: str) -> None:
    picked = llm.resolve(role)
    with _taking:
        card.clear_for(picked.engine, picked.name)
