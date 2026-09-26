import pytest
import requests
from engines import converter
from engines.core import CardState
from engines.drivers import driver
from models.registry import EngineKind
from stand_specs import CONVERTER


class _Reply:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


@pytest.fixture
def supervisor(monkeypatch):
    said = {"body": {"state": "free", "tool": "docling"}, "error": None, "posted": []}

    def get(url, timeout):
        if said["error"]:
            raise said["error"]
        return _Reply(said["body"])

    def post(url, timeout):
        said["posted"].append(url.rsplit("/", 1)[-1])
        return _Reply({"ready": True})

    monkeypatch.setenv("CONVERTER_BASE_URL", "http://converter:5001")
    monkeypatch.setattr(converter.requests, "get", get)
    monkeypatch.setattr(converter.requests, "post", post)
    return said


def test_a_running_tool_holds_the_card_under_its_own_name(supervisor):
    supervisor["body"] = {"state": "holds", "tool": "docling"}
    assert driver(EngineKind.converter).holding(CONVERTER) == ("docling",)


def test_a_supervisor_without_a_child_leaves_the_card_free(supervisor):
    assert driver(EngineKind.converter).holding(CONVERTER) is None
    assert driver(EngineKind.converter).on_card(CONVERTER, "docling") is False


# a silent supervisor may still run its tool, so the card is not read as free
def test_a_silent_supervisor_may_hold_the_card(supervisor):
    supervisor["error"] = requests.Timeout("no answer")
    assert driver(EngineKind.converter).state(CONVERTER) is CardState.UNKNOWN
    assert driver(EngineKind.converter).holding(CONVERTER) == ("converter",)


def test_a_stopped_container_is_down_and_holds_nothing(supervisor):
    supervisor["error"] = requests.ConnectionError("refused")
    assert driver(EngineKind.converter).state(CONVERTER) is CardState.DOWN
    assert driver(EngineKind.converter).holding(CONVERTER) is None


def test_missing_models_read_as_down(supervisor):
    supervisor["body"] = {"state": "down", "missing_models": ["/x.pth"]}
    assert driver(EngineKind.converter).state(CONVERTER) is CardState.DOWN


def test_take_and_let_go_go_to_the_supervisor(supervisor):
    converter_driver = driver(EngineKind.converter)
    assert converter_driver.take_once(CONVERTER, None) is True
    converter_driver.let_go(CONVERTER, ("docling",))
    assert supervisor["posted"] == ["take", "let_go"]


def test_missing_models_refuse_the_take_at_once(supervisor, monkeypatch):
    monkeypatch.setattr(
        converter.requests,
        "post",
        lambda url, timeout: _Reply({"ready": False, "reason": "models missing: /x.pth", "permanent": True}),
    )
    with pytest.raises(converter.WillNotStart, match="models missing"):
        driver(EngineKind.converter).take_once(CONVERTER, None)


def test_a_stop_in_progress_is_waited_out(supervisor, monkeypatch):
    monkeypatch.setattr(
        converter.requests, "post", lambda url, timeout: _Reply({"ready": False, "reason": "a stop is in progress"})
    )
    assert driver(EngineKind.converter).take_once(CONVERTER, None) is False


# the unkillable child is left to the card's own ceiling, which fails loud
def test_a_child_that_survives_let_go_is_only_logged(supervisor, monkeypatch):
    monkeypatch.setattr(
        converter.requests,
        "post",
        lambda url, timeout: _Reply({"held": True, "reason": "child unkillable since 10:00:00"}),
    )
    driver(EngineKind.converter).let_go(CONVERTER, ("docling",))


# it lists no models, so whether it answers is its supervisor's word
def test_a_converter_answers_through_its_supervisor(supervisor):
    from use_cases import stand_health

    assert stand_health.engine_answers(CONVERTER) is True
    supervisor["error"] = requests.ConnectionError("refused")
    assert stand_health.engine_answers(CONVERTER) is False


# two containers speak one word list with no shared code, so the words are held together here
def test_every_state_the_supervisor_says_the_engine_reads():
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "converters" / "supervisor.py").read_text()
    said = set(re.findall(r'"([a-z]+)"', " ".join(line for line in source.splitlines() if "label =" in line)))
    assert said and all(converter._state(word) != converter.CardState.UNKNOWN or word == "unknown" for word in said)
