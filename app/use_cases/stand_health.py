import config
import engines
import gpu
import job_queue
import llm
import logging_setup
import requests
from engines import card as card_holder
from engines import ollama, vllm
from models.jobs import Job
from models.registry import Engine, EngineKind, Model, ModelRole
from orm.sync_db import Session
from sqlalchemy import func, select
from use_cases import search_depth

import db

log = logging_setup.get_logger(__name__)


# from inside the process that owns the card: a probe in a new process sees none of it
def card() -> dict:
    try:
        seen = gpu.memory_mb()
    except Exception as e:  # a probe must not break the route it is read through
        log.warning("stand.card_unreadable", error=str(e))
        return {"cuda": None, "error": str(e)[:120]}
    if seen is None:
        return {"cuda": False}
    free, total = seen
    return {"cuda": True, "free_mb": free, "total_mb": total}


def queue() -> dict:
    with Session() as session:
        counts = dict(
            session.execute(
                select(Job.status, func.count()).group_by(Job.status)
            ).all()
        )
        live = session.scalars(
            select(Job)
            .where(Job.status.in_(job_queue.ACTIVE))
            .order_by(Job.id)
        ).all()
        return {
            "by_status": {str(k): v for k, v in counts.items()},
            "live": [
                {
                    "id": j.id,
                    "type": j.type,
                    "status": j.status,
                    "run_name": (j.options or {}).get("run_name"),
                    "since": j.updated_at.isoformat() if j.updated_at else None,
                }
                for j in live
            ],
        }


# over the union: a role served and never declared drifts as silently as a name that differs
def drifting_roles(declared: dict, served: dict) -> list[str]:
    return sorted(
        role
        for role in set(declared) | set(served)
        if declared.get(role) != served.get(role)
    )


# a role that names no engine sits on the seeded ollama, as bootstrap seats it
def _declared_engine(cfg) -> str:
    if getattr(cfg, "engine", None):
        return cfg.engine
    try:
        return engines.seeded_ollama().name
    except Exception:
        return "the seeded ollama"


# the file declares and the database serves, and bootstrap leaves an assigned role alone
def roles() -> dict:
    # by engine too: `bge-m3` moved to `ollama-cpu` kept its name and read as no drift
    declared = {
        name: f"{cfg.model}@{_declared_engine(cfg)}"
        for name, cfg in config.settings.llm.roles.items()
    }
    with Session() as session:
        rows = session.execute(
            select(ModelRole.role, Model.name, Engine.name)
            .join(Model, Model.id == ModelRole.model_id)
            .join(Engine, Engine.id == Model.engine_id)
        ).all()
    served = {str(role): f"{name}@{engine}" for role, name, engine in rows}
    return {
        "declared": declared,
        "served": served,
        "drift": drifting_roles(declared, served),
    }


def corpus() -> dict:
    variant = config.settings.corpus.variant
    return {
        "active": variant,
        "variants": db.corpus_variants(),
        "fingerprint": db.fingerprint_or_none(variant=variant),
        "sources_missing": db.sources_missing_from(variant=variant),
    }


def depth() -> dict:
    out = {}
    # the variants read here too: a postgres probe bare beside three that are guarded
    for row in db.corpus_variants():
        name = row["variant"]
        try:
            out[name] = search_depth.resolve(name)
        except Exception as e:
            # the name of the failure, not its text: a psycopg message carries the dsn
            out[name] = f"unresolved: {type(e).__name__}"
    return out


# who holds the card, read from the servers; the same body answers `/health` and the MCP tool
def engines_section() -> dict:
    held = card_holder.on_card()
    return {
        "holder": [h.engine.name for h in held],
        "on_card": {h.engine.name: list(h.models) for h in held},
        "vllm_sleeping": {
            spec.name: vllm.is_sleeping(spec) for spec in engines.card_engines(EngineKind.vllm)
        },
        "answers": {spec.name: _answers(spec) for spec in engines.registered()},
    }


# 11.09: the judge's vLLM died of OOM and the stand said nothing; the reranker counts only when used
def roles_down() -> list[str]:
    rerank_used = config.settings.rerank.enabled or config.settings.agent.gate_signal == "cross_encoder"
    down = []
    for role in ("generation", "embedding", "judging", "paraphrasing", "reranking"):
        if role == "reranking" and not rerank_used:
            continue
        spec = llm.resolve(role).engine
        if _answers(spec) is not True:
            down.append(f"{role}: {spec.name} does not answer")
    return down


# a health read waits seconds, not the two minutes a completion may take
def _answers(spec) -> bool | None:
    try:
        return requests.get(f"{engines.base_url(spec)}/v1/models", timeout=3).ok
    except engines.Unconfigured:
        return None
    except Exception:
        return False


# the generator's window from its own engine: `/api/ps` on a vLLM failed the whole stand read
def window() -> dict:
    picked = llm.resolve("generation")
    declared = config.settings.llm.context_length
    if picked.engine.kind is EngineKind.vllm:
        return {"engine": picked.engine.name, "declared": declared, "asked": picked.name,
                "served": vllm.max_model_len(picked.engine, picked.name), "refuses_past_it": True}
    asked = ollama.window_model(picked.name, spec=picked.engine)
    return {"engine": picked.engine.name, "declared": declared, "asked": asked,
            "served": ollama.context_length(asked, picked.engine) if asked else None,
            "refuses_past_it": False}


# every role by its own engine's instrument; only ollama spills, an asleep vLLM is just asleep
def roles_on_card() -> dict:
    seen = {}
    for role in ("generation", "embedding", "judging", "paraphrasing", "reranking"):
        picked = llm.resolve(role)
        spec = picked.engine
        on = card_holder.model_on_card(spec, picked.name)
        seen[role] = {
            "model": picked.name, "engine": spec.name, "placement": str(spec.placement),
            "on_card": on,
            "spilled": spec.kind is EngineKind.ollama and spec.placement in engines.CARD
            and on is False,
        }
    return seen


def stand() -> dict:
    # the card of the generator's engine: with a second ollama a bare ask reads as no residency
    picked = llm.resolve("generation")
    return {
        "card": card(),
        "residency": _or_error("residency", lambda: ollama.residency(picked.engine)
                               if picked.engine.kind is EngineKind.ollama else []),
        "window": _or_error("window", window),
        "roles_on_card": _or_error("roles_on_card", roles_on_card),
        "roles_down": _or_error("roles_down", roles_down),
        "engines": _or_error("engines", engines_section),
        "queue": _or_error("queue", queue),
        "roles": _or_error("roles", roles),
        "corpus": _or_error("corpus", corpus),
        "ef_search": _or_error("ef_search", depth),
    }


# one sick probe must not take the answers the others would still have given
def _or_error(name: str, probe) -> dict:
    try:
        return probe()
    except Exception as e:
        log.warning("stand.probe_failed", probe=name, error=str(e))
        return {"error": str(e)[:120]}
