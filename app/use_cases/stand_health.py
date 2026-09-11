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
from models.registry import Engine, EngineKind, Model, ModelRole, Role
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


# a role that names no engine sits on the seeded ollama; None when that cannot be read
def _declared_engine(cfg) -> str | None:
    if getattr(cfg, "engine", None):
        return cfg.engine
    try:
        return engines.seeded_ollama().name
    except Exception:
        return None


# the file declares and the database serves, and bootstrap leaves an assigned role alone
def roles() -> dict:
    with Session() as session:
        rows = session.execute(
            select(ModelRole.role, Model.name, Engine.name)
            .join(Model, Model.id == ModelRole.model_id)
            .join(Engine, Engine.id == Model.engine_id)
        ).all()
    served = {str(role): engines.label(name, engine) for role, name, engine in rows}
    by_role = {str(role): engine for role, _name, engine in rows}
    # by engine too (`bge-m3` on `ollama-cpu` read as no drift); an unreadable one is compared by name
    declared = {
        name: engines.label(cfg.model, _declared_engine(cfg) or by_role.get(name, "?"))
        for name, cfg in config.settings.llm.roles.items()
    }
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


# every role the enum knows, seated or not: a bare literal list skipped nothing and crashed on one
def _roles() -> list[tuple[str, llm.Resolved | None]]:
    seen = []
    for role in Role:
        try:
            seen.append((role.value, llm.resolve(role.value)))
        except engines.Unnamed:
            seen.append((role.value, None))
    return seen


# the judge's vLLM once died of OOM and the stand said nothing; the reranker counts only when used
def roles_down() -> list[str]:
    from use_cases import card_wait

    rerank_used = card_wait.reranker_needed(agent=True)
    down = []
    for role, picked in _roles():
        if role == Role.reranking and not rerank_used:
            continue
        if picked is None:
            down.append(f"{role}: no model is seated")
        elif _answers(picked.engine) is not True:
            down.append(f"{role}: {picked.engine.name} does not answer")
        elif role == Role.generation and _parserless(picked):
            down.append(f"{role}: {engines.label(picked.name, picked.engine.name)} returns no tool calls")
    return down


# a boot seats the generator unasked; the worker's probe after the wake is what names it here
def _parserless(picked) -> bool:
    spec = picked.engine
    if spec.kind is not EngineKind.vllm:
        return False
    return vllm.known_probe(spec, picked.name, vllm.started_at(spec)) is False


# a health read waits seconds, not the two minutes a completion may take; a paid engine wants its key
def _answers(spec) -> bool | None:
    try:
        headers = {"Authorization": f"Bearer {engines.api_key(spec)}"}
        return requests.get(f"{engines.base_url(spec)}/v1/models", headers=headers, timeout=3).ok
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
    for role, picked in _roles():
        if picked is None:
            # the reranker without its weights on a clean machine: named, and the others still read
            seen[role] = {"model": None, "engine": None, "placement": None, "on_card": None,
                          "spilled": False}
            continue
        spec = picked.engine
        on = card_holder.model_on_card(spec, picked.name)
        seen[role] = {
            "model": picked.name, "engine": spec.name, "placement": str(spec.placement),
            "on_card": on,
            "spilled": card_holder.spilled(spec, picked.name),
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
