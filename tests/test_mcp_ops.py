import asyncio

import mcp_ops
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError


def test_run_metrics_merges_gen_and_retrieval(monkeypatch):
    monkeypatch.setattr(mcp_ops.generation_metrics, "evaluate", lambda rn: {"faithfulness": 7})
    monkeypatch.setattr(mcp_ops.retrieval_metrics, "evaluate", lambda rn: {"hit_at_k": 0.9})
    out = mcp_ops.run_metrics("some_run")
    assert out == {"run_name": "some_run", "faithfulness": 7, "hit_at_k": 0.9}


def test_run_metrics_empty_raises():
    with pytest.raises(ToolError):
        mcp_ops.run_metrics("   ")


def test_compare_runs_empty_raises():
    with pytest.raises(ToolError):
        mcp_ops.compare_runs([])


def test_compare_pools_empty_raises():
    with pytest.raises(ToolError):
        mcp_ops.compare_pools([])


def test_compare_pools_names_the_runs_without_logs(monkeypatch):
    monkeypatch.setattr(mcp_ops, "load_logs", lambda run_name: [])
    with pytest.raises(ToolError) as ei:
        mcp_ops.compare_pools(["ghost"])
    assert "ghost" in str(ei.value)


def test_cancel_job_missing_raises(monkeypatch):
    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a):
            return None

    monkeypatch.setattr(mcp_ops, "Session", lambda: _Session())
    with pytest.raises(ToolError):
        mcp_ops.cancel_job(999999)


def test_run_metrics_masks_unexpected_through_client(monkeypatch):
    def boom(rn):
        raise RuntimeError("SECRET internal detail leaked")

    monkeypatch.setattr(mcp_ops.generation_metrics, "evaluate", boom)

    async def go():
        async with Client(mcp_ops.mcp_ops) as c:
            return await c.call_tool("run_metrics", {"run_name": "x"})

    with pytest.raises(ToolError) as ei:
        asyncio.run(go())
    assert "SECRET" not in str(ei.value)


class _SessionWith:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, _model, _id):
        return self._row


def _experiment():
    from types import SimpleNamespace

    return SimpleNamespace(
        id=36, name="judge_clause_language", kind="rejudge", status="concluded",
        conclusion="the clause that works is the one about meaning",
        results={
            "source_run": "grid_llama31_8b_ru_plain",
            "pairing": "every pair",
            "multiplicity": {"method": "holm", "tests": 4},
            "per_arm": {
                "arm_a": {
                    "arm": {"judge_faithfulness": 3}, "n": 823,
                    "judge": {"model": ["qwen2.5:7b"]},
                    "answers_digest": "sha256:55697f805b010e7d:823",
                    "faithfulness": 7.4,
                }
            },
            "deltas": {
                "repeat_vs_arm_a": {
                    "faithfulness": {"delta": 0.2005, "p": 6.5e-07, "significant_holm": True},
                    "halves": {"faithfulness": {"A": {}, "B": {}}},
                    "same_answers": True,
                }
            },
        },
    )


def test_experiment_results_reads_the_report_without_the_aggregation_scaffolding(monkeypatch):
    # `halves` is the aggregation's control on itself and reads as four more numbers per axis
    monkeypatch.setattr(mcp_ops, "Session", lambda: _SessionWith(_experiment()))
    out = mcp_ops.experiment_results(36)

    assert out["conclusion"].startswith("the clause that works")
    assert out["multiplicity"]["method"] == "holm"
    assert out["arms"]["arm_a"]["answers_digest"].endswith(":823")
    assert "faithfulness" not in out["arms"]["arm_a"], "the arm's own means are not the report"
    assert "halves" not in out["deltas"]["repeat_vs_arm_a"]
    assert out["deltas"]["repeat_vs_arm_a"]["same_answers"] is True


def test_experiment_results_names_the_pairs_it_has_when_asked_for_another(monkeypatch):
    monkeypatch.setattr(mcp_ops, "Session", lambda: _SessionWith(_experiment()))
    with pytest.raises(ToolError, match="repeat_vs_arm_a"):
        mcp_ops.experiment_results(36, pair="nothing_like_it")

    monkeypatch.setattr(mcp_ops, "Session", lambda: _SessionWith(None))
    with pytest.raises(ToolError, match="no experiment"):
        mcp_ops.experiment_results(1)
