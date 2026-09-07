import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from evals import judge_correlation as corr


def test_the_covariate_is_a_share_of_the_answer_found_in_the_contexts():
    contexts = ["the logging middleware records the request method and the url of a request"]
    assert corr.overlap("the logging middleware records the request method", contexts) == 1.0
    assert corr.overlap("Redis persistence is a licensing model for clusters", contexts) == 0.0
    assert corr.overlap("", contexts) == 0.0


def test_a_stratum_measures_how_much_code_a_context_holds():
    # `_is_code_only` was false for all 729 contexts of the first run: a chunk carries a header
    from use_cases.ingest_quality import _is_code_only

    with_header = "[a.md]\n# How do you log a request\n```js\nconst app = express();\n```"
    prose = "In Express.js a middleware handles every incoming request before the route does."

    assert not _is_code_only(with_header), "the header prefix keeps every chunk out of that cut"
    assert corr.code_share([with_header]) > 0.4
    assert corr.code_share([prose]) == 0.0
    assert corr.code_share([]) == 0.0


def test_the_partial_is_the_plain_one_when_the_covariate_explains_nothing():
    assert corr.partial(0.6, 0.0, 0.0) == 0.6
    # a covariate that carries the whole of both leaves nothing behind it
    assert corr.partial(0.64, 0.8, 0.8) == 0.0
    assert corr.partial(None, 0.1, 0.1) is None


def test_a_refusal_is_dropped_by_our_outcome_and_counted_where_a_reader_sees_it(monkeypatch):
    # the standard may give a refusal nan, zero or one, so its own verdict cannot do the cut
    from outcomes import Outcome

    rows = [
        _log(1, faithfulness="7", guest=0.5),
        _log(2, faithfulness="10", guest=None, abstained=True),
        _log(3, faithfulness="10", guest=0.0),
        _log(4, faithfulness="7", guest=0.5, marked=[]),
        _log(5, faithfulness="7", guest=0.5),
    ]
    monkeypatch.setattr(corr, "load_logs", lambda run=None: rows)
    outcomes = {3: Outcome.refused, 5: Outcome.narrated_call}
    monkeypatch.setattr(corr, "_outcome", lambda ql: outcomes.get(ql.id, Outcome.answered))
    kept, counts = corr.rows_of()

    assert [r["id"] for r in kept] == [1]
    assert counts["refused_excluded"] == 1
    assert counts["guest_abstained"] == 1
    # a row of another pool and a narrated call are both outside the declared population
    assert counts["outside_the_declared_population"] == 2


def test_the_report_says_which_prediction_held():
    rows = [
        {"id": i, "run_name": "r", "pipeline": "agent", "ours": 0.7, "guest": 0.7,
         "overlap": i / 10, "code_share": 0.0, "on_card": True,
         "guest_precision": None, "guest_recall": None}
        for i in range(1, 11)
    ]
    out = corr.report(rows, {"refused_excluded": 0, "guest_abstained": 0})
    assert out["schema"] == corr.SCHEMA
    assert out["predictions"]["n_at_least_100"] is False
    assert out["n"] == 10


def _log(id_, faithfulness, guest, abstained=False, marked=("a.md",)):
    entry = {"score": guest, "abstained": abstained}
    return SimpleNamespace(
        id=id_, run_name="r", pipeline="agent", faithfulness=faithfulness,
        answer="an answer about middleware", contexts=["a context about middleware"],
        question=SimpleNamespace(marked_sources=list(marked)),
        metrics={"ragas_faithfulness": entry},
    )


def test_the_tool_reads_the_report_rather_than_computing_a_second_one():
    # `run_metrics` and `compare_pools` are tools without routes: the ops surface is the door
    import inspect

    import mcp_ops

    source = inspect.getsource(mcp_ops.judge_correlation_report)
    assert "judge_correlation.report(" in source and "judge_correlation.rows_of(" in source


def test_a_probe_reports_an_interval_and_names_its_instrument(monkeypatch):
    # a point estimate on ten rows was read as a refutation once, and the interval said otherwise
    from evals import guest_llm
    from evals import guest_probes as probes

    # the stamp reads the role from the base, and a unit test of the report shape has no base
    monkeypatch.setattr(guest_llm.llm, "resolve_name", lambda role: "q:7b")

    done = [
        {"row": i, "arm": "copy", "score": 1.0, "n_statements": 2, "error": None, "overlap": 1.0}
        for i in range(5)
    ] + [
        {"row": i, "arm": "negated", "score": 0.5, "n_statements": 2, "error": None,
         "overlap": 0.6}
        for i in range(5)
    ]
    out = probes.report("negated", done)

    assert out["schema"] == probes.SCHEMA
    assert set(out["instrument"]) >= {"ragas", "model", "role", "runtime"}
    assert out["paired"]["mean"] == 0.5 and out["paired"]["n"] == 5
    assert out["paired"]["ci95"] == [0.5, 0.5]
    assert probes.interval([1.0]) is None


def test_the_language_probe_pairs_one_row_across_two_languages():
    # the guest does not penalise Russian; whether ours does is three links away, and this is one
    from evals import judge_language as jl

    rows = [
        {"row": 1, "lang": "en", "score": 10}, {"row": 1, "lang": "ru", "score": 7},
        {"row": 2, "lang": "en", "score": 9}, {"row": 2, "lang": "ru", "score": 9},
    ]
    out = jl.report(rows)

    assert out["schema"] == jl.SCHEMA
    assert out["means"] == {"en": 9.5, "ru": 8.0}
    assert out["paired"]["n"] == 2 and out["paired"]["mean"] == 1.5
    assert out["paired"]["moved"] == 1


def test_rho_carries_the_band_the_record_quoted_without_one():
    # 0.500 [0.370, 0.614] lived in the notes and in no file the script could reproduce
    from evals import judge_correlation as jc

    rows = [{"ours": i / 10, "guest": i / 10, "overlap": 0.1, "code_share": 0.0}
            for i in range(1, 11)]
    low, high = jc.rho_ci(rows, "ours", "guest")

    assert 0.9 <= low <= high <= 1.0
    assert jc.rho_ci(rows[:2], "ours", "guest") is None


def test_the_language_probe_is_a_job_and_a_door_and_one_loop(monkeypatch):
    # three runs of a script beside the queue is an operation, and it needed a waiter each time
    import inspect
    from pathlib import Path

    from evals import judge_language as jl
    from job_handlers import judging
    from job_handlers.base import HANDLERS

    assert "judge_language" in HANDLERS
    assert judging.judge_language is HANDLERS["judge_language"]
    assert "measure" in inspect.getsource(judging.judge_language), "the job re-implements the loop"
    script = (Path(__file__).resolve().parent.parent / "scripts" / "judge_language_probe.py")
    assert "measure(" in script.read_text(), "the script re-implements the loop"
    assert callable(jl.measure)


def test_a_job_writes_its_number_where_it_can_be_cited(tmp_path, monkeypatch):
    # removing the caller's `out` closed a hole and left the number in a log line only
    from evals import measurements

    monkeypatch.setattr(measurements, "FOLDER", tmp_path)
    where = measurements.record(
        "judge_language", "arc3/agent baseline", {"n_rows": 108}, on=date(2026, 9, 7)
    )

    assert Path(where).name == "judge_language_arc3_agent_baseline_20260907.json"
    assert json.loads(Path(where).read_text())["n_rows"] == 108
    # a name that a caller could aim at a path of its own choosing
    assert "/" not in measurements._slug("../../etc/passwd")
