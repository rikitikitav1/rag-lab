import config
import llm
import logging_setup
import version
from engines import ollama
from models.registry import Role

import db

log = logging_setup.get_logger(__name__)

# 5 engine per role; 6 what the engine refused; 7 that field renamed `engine_refused`
SCHEMA = 7

# every key a run records about how it was configured, written whether or not it applies
KEYS = (
    "schema",
    "rerank",
    "rerank_device",
    "distance_threshold",
    "k",
    "phased",
    "variant",
    "keyword",
    "ef_search",
    "variant_policy",
    "corpus",
    "corpus_fingerprint",
    "code_version",
    "context_length",
    "orchestrator",
    "fallback_policy",
    "gate",
    "drop_weak_context",
    "topic",
    "max_hops",
    "truncated_hops",
    "mcp",
    "mcp_configured",
    # the system prompt carries a language directive, and a replay could not rebuild it
    "language",
    # an arm that says the tools again after a tool answer must say so, or it is silently a new arm
    "restate_tools",
    # per role, and read by the role keys: a bare {} means no role was read, not "one engine"
    "engines",
    # what a role asked of its engine and the engine would not carry: never read as applied
    "engine_refused",
)


# the two roles a run answers with; judging is stamped per row, where the bench can override it
ANSWERING = (Role.generation, Role.embedding)


# a report must not die on an unreachable registry: the engine is extra, the run is the record
def _by_role(picked) -> tuple[dict, dict]:
    named, dropped = {}, {}
    for role in ANSWERING:
        try:
            spec = picked.engine if role is Role.generation else llm.resolve(role).engine
            if spec is None:
                continue
            named[role] = spec.name
            dropped[role] = llm.sampler(role, spec).dropped
        except Exception as e:
            log.warning("run_snapshot.engine_unread", role=role, error=str(e))
    return named, dropped


# the comment below promises the report survives an unreadable registry, so this one does too
def _generator(model: str | None):
    try:
        return llm.resolve_for(Role.generation, model)
    except Exception as e:
        log.warning("run_snapshot.generator_unread", model=model, error=str(e))
        return llm.Resolved(model or "?", None)


def _rerank_device() -> str | None:
    try:
        import rerank

        return rerank.device()
    except Exception:
        return None


# an absent key and a null read the same to every later reader, and one of them is honest
def of_run(
    *,
    variant: str,
    use_rerank,
    k,
    ef_search,
    distance_threshold,
    model=None,
    rerank_device=None,
    **filled,
) -> dict:
    unknown = sorted(set(filled) - set(KEYS))
    if unknown:
        raise ValueError(f"the run snapshot has no place for {unknown}")
    picked = _generator(model)
    named, dropped = _by_role(picked)
    common = {
        "schema": SCHEMA,
        "rerank": use_rerank,
        "rerank_device": (rerank_device or _rerank_device()) if use_rerank else None,
        "distance_threshold": distance_threshold,
        "k": k,
        "variant": variant,
        "keyword": config.keyword_switches(),
        "ef_search": ef_search,
        # a variant in the table and absent from the config is possible, and raising kills an answer
        "variant_policy": config.settings.corpus.policy_or_none(variant),
        "corpus": config.settings.corpus.description,
        "corpus_fingerprint": db.fingerprint_or_none(variant=variant),
        # the commit both pipelines ran, so two arms can be shown to have run the same code
        "code_version": version.CODE_VERSION,
        "context_length": ollama.context_length(picked.name, picked.engine),
        "engines": named,
        "engine_refused": dropped,
    }
    return {key: None for key in KEYS} | common | filled


# single_shot records rows returned, `config.k` in 9414 of 9415; the agent records sources
RETRIEVAL_KEYS = ("results_count", "min_distance", "top_rerank_score", "dropped_sources")
# `question-log?max_distance` filters on this, so the two pipelines must round it alike
DISTANCE_DIGITS = 3


def of_retrieval(**filled) -> dict:
    unknown = sorted(set(filled) - set(RETRIEVAL_KEYS))
    if unknown:
        raise ValueError(f"the retrieval snapshot has no place for {unknown}")
    if filled.get("min_distance") is not None:
        filled["min_distance"] = round(filled["min_distance"], DISTANCE_DIGITS)
    return {key: None for key in RETRIEVAL_KEYS} | filled


# the preflight unpacks `policy` out of this: a rename in a literal made it compare nothing
def of_topic(threshold, score, policy) -> dict:
    return {
        "threshold": threshold,
        "score": round(score, 3) if score is not None else None,
        "input": "question",
        # what was configured, beside what was applied: the applied number varies by language
        "policy": policy,
    }
