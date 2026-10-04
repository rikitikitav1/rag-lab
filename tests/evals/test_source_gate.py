# a source's own questions read clamped and open: two numbers, two next steps, and nothing read under the floor
import config
import pytest
from use_cases import source_gate


@pytest.fixture
def gate(monkeypatch):
    written = {}

    class _Row:
        raw = {"verdict": "ok"}

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalar(self, stmt):
            return written.setdefault("row", _Row())

        def commit(self):
            return None

    class _Conn(_Session):
        pass

    monkeypatch.setattr(source_gate, "Session", _Session)
    monkeypatch.setattr(source_gate.engine, "connect", lambda: _Conn())
    monkeypatch.setattr(source_gate.search_depth, "resolve", lambda variant=None, ef=None: 200)
    monkeypatch.setattr(config.settings.intake.quality, "source_gate",
                        config.SourceGateCfg(clamped_min=0.85, open_min=0.75, min_questions=4))

    def ranks(opened, clamped):
        def measure(searcher, conn, set_name, variant, limit, exact, ef=None, source=None, clamped=False):
            return [{"file_rank": r} for r in (clamped_ranks if clamped else opened_ranks)]
        opened_ranks, clamped_ranks = opened, clamped
        monkeypatch.setattr(source_gate.retrieval_compare, "measure", measure)

    return ranks, written


def test_a_source_found_in_itself_but_not_in_the_corpus_waits_for_the_owner(gate):
    ranks, written = gate
    ranks([1, 9, None, 2], [1, 1, 2, 3])
    said = source_gate.run("book", "set")
    assert (said["verdict"], said["clamped"], said["open"]) == ("open_low", 1.0, 0.5)
    assert written["row"].raw["gate"]["verdict"] == "open_low" and written["row"].raw["verdict"] == "ok"


def test_a_low_clamped_number_goes_to_the_agent_and_a_short_set_reads_nothing(gate):
    ranks, _ = gate
    ranks([1, 1, 1, 1], [1, None, None, 1])
    assert source_gate.run("book", "set")["verdict"] == "clamped_low"
    ranks([1, 1], [1, 1])
    said = source_gate.run("book", "set")
    assert said["verdict"] == "too_few" and "clamped" not in said
