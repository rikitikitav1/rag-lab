import logging_setup
import requests

from .core import CardState, EngineSpec, Unconfigured, base_url

log = logging_setup.get_logger(__name__)

HTTP_TIMEOUT = 10
# the supervisor kills a stubborn tool after 20 s and waits 20 s more
LET_GO_TIMEOUT = 45


# the supervisor runs in its own image and says its state in the stand's words; any other word is unknown
def _state(word) -> CardState:
    try:
        return CardState(word)
    except ValueError:
        return CardState.UNKNOWN


def _url(spec: EngineSpec, path: str) -> str:
    return base_url(spec) + path


# the supervisor's own words: the tool it runs, since when, and why it will not start
def reading(spec: EngineSpec) -> tuple[CardState, dict]:
    try:
        seen = requests.get(_url(spec, "/supervisor"), timeout=HTTP_TIMEOUT)
        seen.raise_for_status()
        body = seen.json()
        return _state(body.get("state")), body
    except requests.Timeout as e:
        # a ConnectTimeout is a ConnectionError too, and a silent supervisor may still run its tool
        log.warning("converter.state_unknown", engine=spec.name, error=str(e))
        return CardState.UNKNOWN, {}
    except (requests.ConnectionError, Unconfigured):
        return CardState.DOWN, {}
    except Exception as e:
        log.warning("converter.state_unknown", engine=spec.name, error=str(e))
        return CardState.UNKNOWN, {}


def card_state(spec: EngineSpec) -> CardState:
    return reading(spec)[0]


class WillNotStart(RuntimeError):
    pass


# one attempt: the child starts on the first, and the card polls until it answers ready
def take_once(spec: EngineSpec) -> bool:
    seen = requests.post(_url(spec, "/supervisor/take"), timeout=HTTP_TIMEOUT)
    seen.raise_for_status()
    body = seen.json()
    # missing models will not appear while the card waits, so the refusal is said at once
    if body.get("permanent"):
        raise WillNotStart(f"{spec.name}: {body.get('reason')}")
    if body.get("reason"):
        log.warning("converter.take_refused", engine=spec.name, reason=body["reason"])
    return bool(body.get("ready"))


def let_go(spec: EngineSpec) -> None:
    seen = requests.post(_url(spec, "/supervisor/let_go"), timeout=LET_GO_TIMEOUT)
    seen.raise_for_status()
    body = seen.json()
    if body.get("held"):
        log.warning("converter.still_held", engine=spec.name, reason=body.get("reason"))
