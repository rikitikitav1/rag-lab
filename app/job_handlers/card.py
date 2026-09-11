import threading

import engines
import llm
from engines import card, ollama
from models.registry import EngineKind

from .base import register


# the one road to the card: loads, wakes and sleeps happen here and nowhere else in the stand
@register("hand_card")
def hand_card(options: dict) -> None:
    target = engines.spec_of_id(options["engine_id"])
    if target is None:
        raise ValueError(f"engine {options['engine_id']} is not registered")
    card.hand_to(target, options.get("model"))


_taking = threading.Lock()


# the preamble takes the card for a job's own role; a second role on another engine takes it here
def take_for_call(spec, _model: str) -> None:
    if spec.placement not in engines.CARD:
        return
    with _taking:
        if card.holds_for(spec):
            return
        # no model: ollama loads on the call itself, and an embedder cannot be loaded by a generate
        card.hand_to(spec)


# 11.09: a batch of 64 chunks beside a resident 8b dropped ollama's runner with CUDA OOM
def clear_the_engine_for(role: str) -> None:
    picked = llm.resolve(role)
    spec = picked.engine
    if spec.kind is not EngineKind.ollama or spec.placement not in engines.CARD:
        return
    keep = {picked.name, f"{picked.name}:latest"}
    for resident in ollama.residency(spec):
        if resident["model"] not in keep:
            ollama.unload(resident["model"], spec)
