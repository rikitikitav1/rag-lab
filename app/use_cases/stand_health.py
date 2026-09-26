import config
import engines
import gpu
import job_queue
import llm
import logging_setup
import version
from engines import card as card_holder
from engines import ollama, vllm
from models.jobs import Job
from models.registry import SAMPLING_ROLES, Engine, EngineKind, Model, ModelRole, Role
from orm.sync_db import Session
from sqlalchemy import func, select
from use_cases import search_depth

import db

log = logging_setup.get_logger(__name__)


# the driver's reading of the whole card, every process on it counted
def card() -> dict:
    try:
        seen = gpu.memory_mb()
    except Exception as e:  # a probe must not break the route it is read through
        log.warning("stand.card_unreadable", error=str(e))
        return {"cuda": None, "error": type(e).__name__}
    if seen is None:
        return {"cuda": False}
    free, total = seen
    return {"cuda": True, "free_mb": free, "total_mb": total}


def queue() -> dict:
    with Session() as session:
        counts = dict(session.execute(select(Job.status, func.count()).group_by(Job.status)).all())
        live = session.scalars(select(Job).where(Job.status.in_(job_queue.ACTIVE)).order_by(Job.id)).all()
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
    return sorted(role for role in set(declared) | set(served) if declared.get(role) != served.get(role))


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
        # in words: an empty holder list read as a fault on a host that has no card at all
        "summary": (
            f"the card is held by {', '.join(h.engine.name for h in held)}" if held else "no engine holds the card"
        ),
        "on_card": {h.engine.name: list(h.models) for h in held},
        "vllm_sleeping": {spec.name: vllm.is_sleeping(spec) for spec in engines.card_engines(EngineKind.vllm)},
        "answers": {spec.name: engine_answers(spec) for spec in engines.registered()},
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
            # an optional role nobody seated is not down; the reranker stops being optional once used
            if role in config.REQUIRED_ROLES or role == Role.reranking:
                down.append(f"{role}: no model is seated")
        elif engine_answers(picked.engine) is not True:
            down.append(f"{role}: {picked.engine.name} does not answer{_no_card_hint(picked.engine)}")
        # an ollama that lost the card still answers, and loads its models on the processor
        elif card_holder.spilled(picked.engine, picked.name):
            down.append(
                f"{role}: {engines.label(picked.name, picked.engine.name)} is not whole on the card;"
                " a container that lost the card answers so (docs/stand_modes.md)"
            )
        elif role == Role.generation and _parserless(picked):
            down.append(f"{role}: {engines.label(picked.name, picked.engine.name)} returns no tool calls")
    return down


# an engine of the card that is down is often a host with no card, and the way out is a mode
def _no_card_hint(spec) -> str:
    if spec.placement not in engines.CARD:
        return ""
    # already without a card, the way out is the seat and not the mode
    if config.CONFIG_OVERLAY:
        return "; this stand runs without a card: seat the role on `ollama-cpu` with `PUT /v1/role`"
    return "; a host without a card runs `scripts/up.sh --cpu` (docs/stand_modes.md)"


# a boot seats the generator unasked; the worker's probe after the wake is what names it here
def _parserless(picked) -> bool:
    spec = picked.engine
    if spec.kind is not EngineKind.vllm:
        return False
    return vllm.probe_now(spec, picked.name) is False


# one answer for every door, in seconds: an engine with no address or key is unconfigured (None), not down
def engine_answers(spec) -> bool | None:
    try:
        reading = engines.driver(getattr(spec, "kind", None))
        if not reading.serves_models:
            return reading.state(spec) not in engines.SILENT
        return engines.served_models(spec) is not None
    except engines.Unconfigured:
        return None
    except Exception:
        return False


# the generator's window from its own engine: `/api/ps` on a vLLM failed the whole stand read
def window() -> dict:
    picked = llm.resolve("generation")
    reading = engines.driver(picked.engine.kind)
    asked = reading.window_model(picked.engine, picked.name)
    return {
        "engine": picked.engine.name,
        "declared": config.settings.llm.context_length,
        "asked": asked,
        "served": reading.window(picked.engine, asked) if asked else None,
        "refuses_past_it": reading.refuses_past_the_window,
    }


# every role by its own engine's instrument; only ollama spills, an asleep vLLM is just asleep
def roles_on_card() -> dict:
    seen = {}
    for role, picked in _roles():
        if picked is None:
            # the reranker without its weights on a clean machine: named, and the others still read
            seen[role] = {"model": None, "engine": None, "placement": None, "on_card": None, "spilled": False}
            continue
        spec = picked.engine
        on = card_holder.model_on_card(spec, picked.name)
        seen[role] = {
            "model": picked.name,
            "engine": spec.name,
            "placement": str(spec.placement),
            "on_card": on,
            "spilled": card_holder.spilled_reading(spec, on),
        }
    return seen


# declared once and held in each ollama model of a role: a model pulled by hand or recreated shows here
def repetition_penalty() -> dict:
    served = {}
    for role, picked in _roles():
        if picked is None or picked.engine.kind is not EngineKind.ollama or Role(role) not in SAMPLING_ROLES:
            continue
        served[engines.label(picked.name, picked.engine.name)] = ollama.repetition_penalty_served(
            picked.name, picked.engine
        )
    declared = config.settings.llm.repetition_penalty
    return {"declared": declared, "served": served, "drift": sorted(n for n, v in served.items() if v != declared)}


# what each seated role's calls send: a model row overriding its role is the design, not drift
def samplers() -> dict:
    out = {}
    for role, picked in _roles():
        if picked is None or role not in config.settings.llm.roles:
            continue
        out[role] = {
            "sampler": llm.sampler(role, picked).sent,
            "overrides": sorted(k for k in (picked.options or {}) if k in engines.SAMPLER_KEYS),
        }
    return out


# a worker that claims nothing looks from the queue exactly like a worker with nothing to do
def code() -> dict:
    said = version.what_the_worker_loaded()
    on_disk = version.tree_stamp()
    out = {"on_disk": on_disk, "api_loaded": version.LOADED_TREE, "code_version": version.CODE_VERSION}
    if said is None:
        return out | {"worker": "has not said which code it loaded"}
    out |= {
        "worker_loaded": said.get("stamp"),
        "worker_said_at": said.get("at"),
        # hygiene, as in a run's snapshot: the tree moved beside the worker, which may not have imported it
        "worker_tree_moved": said.get("stamp") != on_disk,
    }
    # the reading that decides, the same one `compare.code_by_run` uses: a file the worker imported moved
    if "loaded_differs" not in said:
        return out | {
            "worker_loaded_moved": None,
            "too_old_to_tell": "the worker's stamp predates the loaded-files reading",
        }
    return out | {"worker_loaded_moved": said["loaded_differs"]}


def stand() -> dict:
    # the card of the generator's engine: with a second ollama a bare ask reads as no residency
    def residency() -> list:
        picked = llm.resolve("generation")
        return ollama.residency(picked.engine) if picked.engine.kind is EngineKind.ollama else []

    # an unseated generator answered 500 here, on the page that exists to say so
    return {
        "card": card(),
        "residency": _or_error("residency", residency),
        "window": _or_error("window", window),
        "repetition_penalty": _or_error("repetition_penalty", repetition_penalty),
        "roles_on_card": _or_error("roles_on_card", roles_on_card),
        "roles_down": _or_error("roles_down", roles_down),
        "engines": _or_error("engines", engines_section),
        "queue": _or_error("queue", queue),
        "roles": _or_error("roles", roles),
        "samplers": _or_error("samplers", samplers),
        "corpus": _or_error("corpus", corpus),
        "ef_search": _or_error("ef_search", depth),
        "code": _or_error("code", code),
    }


# one sick probe must not take the answers the others would still have given
def _or_error(name: str, probe) -> dict:
    try:
        return probe()
    except Exception as e:
        log.warning("stand.probe_failed", probe=name, error=str(e))
        # the name of the failure, not its text: the route answers without a key
        return {"error": type(e).__name__}
