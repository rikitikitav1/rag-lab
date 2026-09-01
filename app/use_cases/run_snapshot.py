import config
import llm
import version

import db

# 1 is every row written before this key existed; 2 writes every key it models, always
SCHEMA = 2

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
)


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
        "context_length": llm.server_context_length(model or llm.resolve_name("generation")),
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
