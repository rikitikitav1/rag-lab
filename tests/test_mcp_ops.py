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
