import config
import logging_setup
import requests
from models.registry import EngineKind

from . import ollama, vllm
from .core import CardState, EngineSpec, NotSupported, Unconfigured
from .lookup import registered

log = logging_setup.get_logger(__name__)


# what one kind of engine does its own way; the card, the pull and the delete ask it, not the kind
class Driver:
    # what the card reading is called, for the reader of a null
    instrument: str | None = None
    # what a handover waits for, named in the refusal when it never comes
    takes_the_card = "take the card"
    # a model may answer half on the processor, with other kernels
    spills = False
    # it loads a role on the first call, so a card nobody else holds is already its own
    owns_a_free_card = False
    # a prompt past the window is refused rather than cut
    refuses_past_the_window = False

    def state(self, spec: EngineSpec) -> CardState | None:
        return None

    def holding(self, spec: EngineSpec) -> tuple[str, ...] | None:
        return None

    def on_card(self, spec: EngineSpec, model: str) -> bool | None:
        return None

    def let_go(self, spec: EngineSpec, models: tuple[str, ...]) -> None:
        return None

    # one model out of memory; a server that holds one model for its life has nothing to give back
    def unload(self, spec: EngineSpec, model: str) -> None:
        return None

    # one attempt; the card polls it under its own ceiling
    def take_once(self, spec: EngineSpec, model: str | None) -> bool:
        return True

    def load_off_card(self, spec: EngineSpec, model: str | None) -> None:
        return None

    def make_room(self, spec: EngineSpec, keep: str) -> None:
        return None

    def window(self, spec: EngineSpec, model: str) -> int | None:
        return None

    def window_model(self, spec: EngineSpec, configured: str | None) -> str | None:
        return configured

    def weights_store(self) -> str | None:
        return None

    def expected_size(self, name: str) -> int | None:
        return None

    def pull(self, name: str, spec: EngineSpec) -> None:
        raise NotSupported(f"{spec.name} keeps its weights itself: nothing to pull here")

    def artifact(self, name: str, spec: EngineSpec) -> dict:
        return {}

    def refuse_delete(self, name: str) -> None:
        return None

    def delete(self, name: str, spec: EngineSpec) -> None:
        raise NotSupported(f"{spec.name} keeps its weights itself: nothing to delete here")

    def weights_key(self, name: str) -> str:
        return name

    # the window a model gets once it loads, for a model nobody has loaded yet
    def configured_window(self, spec: EngineSpec) -> int | None:
        return None


class Ollama(Driver):
    instrument = "ollama /api/ps"
    takes_the_card = "load"
    spills = True
    owns_a_free_card = True

    def state(self, spec: EngineSpec) -> CardState:
        return ollama.card_state(spec)

    # the same rule as for vLLM: a silence may hold the card, a stopped server does not
    def holding(self, spec: EngineSpec) -> tuple[str, ...] | None:
        state, seen = ollama.card_reading(spec)
        if state not in (CardState.HOLDS, CardState.UNKNOWN):
            return None
        return tuple(ollama.on_card_models(seen))

    def on_card(self, spec: EngineSpec, model: str) -> bool | None:
        wanted = ollama.spellings(model)
        seen = [m for m in ollama.residency(spec) if m["model"] in wanted]
        return seen[0]["vram_mb"] >= seen[0]["size_mb"] if seen else None

    def let_go(self, spec: EngineSpec, models: tuple[str, ...]) -> None:
        for name in models:
            ollama.unload(name, spec)

    def unload(self, spec: EngineSpec, model: str) -> None:
        ollama.unload(model, spec)

    def take_once(self, spec: EngineSpec, model: str | None) -> bool:
        if model:
            ollama.load_into_memory(model, spec)
        return True

    def load_off_card(self, spec: EngineSpec, model: str | None) -> None:
        if model:
            ollama.load_into_memory(model, spec)

    # a batch of 64 chunks beside a resident 8b dropped ollama's runner with CUDA OOM
    def make_room(self, spec: EngineSpec, keep: str) -> None:
        for resident in ollama.residency(spec):
            if resident["model"] not in ollama.spellings(keep):
                ollama.unload(resident["model"], spec)

    def window(self, spec: EngineSpec, model: str) -> int | None:
        return ollama.context_length(model, spec)

    # compose starts ollama with `OLLAMA_CONTEXT_LENGTH` from the same default
    def configured_window(self, spec: EngineSpec) -> int | None:
        return config.settings.llm.context_length

    def window_model(self, spec: EngineSpec, configured: str | None) -> str | None:
        return ollama.window_model(configured, spec=spec)

    def expected_size(self, name: str) -> int | None:
        return ollama.registry_size(name)

    def pull(self, name: str, spec: EngineSpec) -> None:
        ollama.pull_model(name, spec)

    def artifact(self, name: str, spec: EngineSpec) -> dict:
        return ollama.artifact_of(name, spec)

    def delete(self, name: str, spec: EngineSpec) -> None:
        ollama.delete_model(name, spec)

    def weights_key(self, name: str) -> str:
        return ollama.add_tags([name])[0]


class Vllm(Driver):
    instrument = "vllm /is_sleeping"
    takes_the_card = "wake"
    refuses_past_the_window = True

    def state(self, spec: EngineSpec) -> CardState:
        return vllm.card_state(spec)

    # silence is not a free card: an unanswered server may be awake on it
    def holding(self, spec: EngineSpec) -> tuple[str, ...] | None:
        if vllm.card_state(spec) not in (CardState.AWAKE, CardState.UNKNOWN):
            return None
        try:
            return tuple(vllm.served(spec))
        except Exception:
            return ()

    def on_card(self, spec: EngineSpec, model: str) -> bool | None:
        seen = {CardState.AWAKE: True, CardState.ASLEEP: False}.get(vllm.card_state(spec))
        if not seen:
            return seen
        # awake is not enough: the server must serve this very model
        try:
            return model in vllm.served(spec)
        except Exception:
            return None

    def let_go(self, spec: EngineSpec, models: tuple[str, ...]) -> None:
        vllm.sleep(spec)

    # a wake on a card not yet free fails and leaves the server asleep; a later one succeeds
    def take_once(self, spec: EngineSpec, model: str | None) -> bool:
        if vllm.is_sleeping(spec) is False:
            return True
        try:
            vllm.wake_up(spec)
            return True
        # a server that dies mid-wake answers with a connection error, and the ceiling names it
        except (vllm.WakeFailed, requests.RequestException) as e:
            log.info("card.wake_retry", engine=spec.name, error=str(e))
            return False

    def window(self, spec: EngineSpec, model: str) -> int | None:
        return vllm.max_model_len(spec, model)

    def weights_store(self) -> str | None:
        return vllm.weights_cache()

    def expected_size(self, name: str) -> int | None:
        return vllm.repo_size(name)

    def pull(self, name: str, spec: EngineSpec) -> None:
        vllm.pull_weights(name)

    def artifact(self, name: str, spec: EngineSpec) -> dict:
        return vllm.artifact_of(name)

    # every vLLM service reads the one host cache, so each one is asked, and silence is not a no
    def refuse_delete(self, name: str) -> None:
        for spec in registered():
            if spec.kind is not EngineKind.vllm:
                continue
            try:
                names = vllm.served(spec)
            # first: a ConnectTimeout is a ConnectionError too, and a silent host may serve it
            except requests.Timeout as e:
                raise vllm.StillServed(_unknown(spec, name, e)) from e
            except (requests.ConnectionError, Unconfigured):
                continue
            except Exception as e:
                raise vllm.StillServed(_unknown(spec, name, e)) from e
            if name in names:
                raise vllm.StillServed(f"{name} is served by {spec.name} right now; stop that server first")

    def delete(self, name: str, spec: EngineSpec) -> None:
        vllm.delete_weights(name)


def _unknown(spec: EngineSpec, name: str, e: Exception) -> str:
    return f"{spec.name} did not answer, so whether it serves {name} is unknown: {e}"


_DRIVERS = {EngineKind.ollama: Ollama(), EngineKind.vllm: Vllm()}
# a paid engine: no card, no weights here, and every question about them answers nothing
_REMOTE = Driver()


def driver(kind: EngineKind | None) -> Driver:
    return _DRIVERS.get(kind, _REMOTE)


# the server's window when the model is loaded, else the one it will load with: a guard on either passed unloaded models
def window_or_configured(spec: EngineSpec, model: str) -> int | None:
    reading = driver(spec.kind)
    try:
        window = reading.window(spec, model)
    except Exception as e:
        log.warning("engine.window_unread", engine=spec.name, model=model, error=str(e))
        window = None
    return window or reading.configured_window(spec)
