import engines
from models.registry import EngineKind, Placement

# the stand's engines as the seed registers them, for tests that need a spec and not a database
OLLAMA = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
VLLM = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
OLLAMA_CPU = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
VLLM_RERANK = engines.EngineSpec(7, "vllm-rerank", EngineKind.vllm, "VLLM_RERANK", Placement.gpu)


# the whole row a queued job answers with, as a door reads it back after the commit
def queued_job(type: str = "eval_run", options: dict | None = None, id: int = 1):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    now = datetime.now(UTC)
    return SimpleNamespace(
        id=id,
        type=type,
        options=options or {},
        queue="default",
        status="new",
        error=None,
        elapsed=None,
        apply_since=now,
        created_at=now,
        updated_at=now,
    )


CONVERTER = engines.EngineSpec(9, "converter", EngineKind.converter, "CONVERTER", Placement.gpu)
