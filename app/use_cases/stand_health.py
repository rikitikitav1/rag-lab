import config
import gpu
import job_queue
import llm
import logging_setup
from models.jobs import Job
from models.registry import Model, ModelRole
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


# the file declares and the database serves, and bootstrap leaves an assigned role alone
def roles() -> dict:
    declared = {name: cfg.model for name, cfg in config.settings.llm.roles.items()}
    with Session() as session:
        rows = session.execute(
            select(ModelRole.role, Model.name).join(Model, Model.id == ModelRole.model_id)
        ).all()
    served = {str(role): name for role, name in rows}
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


def stand() -> dict:
    # one rule with the preflight and the record: the route used to name another model
    loaded = llm.residency()
    asked = llm.window_model(loaded)
    return {
        "card": card(),
        "residency": loaded,
        "window": {
            "declared": config.settings.llm.context_length,
            "asked": asked,
            "served": llm.server_context_length(asked) if asked else None,
        },
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
