import config
import engines
import llm
import logging_setup
import version
from engines import answer_parsers, card
from errors import StandFault
from models.registry import SAMPLING_ROLES, Role

import db

log = logging_setup.get_logger(__name__)

# the snapshot's shape, raised with every key it gains
SCHEMA = 13

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
    # per role, what the engine set on its own: a vLLM and an ollama arm may run one model in two dtypes
    "engine_added",
    # per role, what went out: a budget set on the model changes the run's ruler, and the record must say
    "samplers",
    # per role, read from the server: a generator half on the cpu answered with other kernels
    "on_card",
    # per role, the parser that cut its answers: a broker writes thinking and call markup into the text
    "answer_parsers",
    # per cloud role, what keyed the broker's cache: a second pass without it could have read the first
    "cache_keys",
)


# the two roles a run answers with; judging is stamped per row, where the bench can override it
ANSWERING = (Role.generation, Role.embedding)


# a report must not die on an unreachable registry: the engine is extra, the run is the record
def _by_role(picked, roles=ANSWERING) -> tuple[dict, dict, dict, dict, dict, dict]:
    named, samplers, placed, added, cache_keys, parsers = {}, {}, {}, {}, {}, {}
    for role in roles:
        try:
            chosen = picked if role is Role.generation else model_of(role)
            spec = chosen.engine
            if spec is None:
                continue
            named[role] = spec.name
            samplers[role] = llm.sampler(role, chosen)
            placed[role] = card.model_on_card(spec, chosen.name)
            added[role] = {key: value for key, value in engines.added_by(spec, chosen.name).items()
                           if role in SAMPLING_ROLES or key != "repetition_penalty"}
            if key := llm.cache_key_of(spec):
                cache_keys[role] = key
            parsers[role] = answer_parsers.label(getattr(chosen, "parser", answer_parsers.NONE))
        except Exception as e:
            log.warning("run_snapshot.engine_unread", role=role, error=str(e))
    return named, samplers, placed, added, cache_keys, parsers


# by the role's own engine, and a failed read is unknown rather than a reason to stop the run
def placed(role: str) -> bool | None:
    try:
        picked = model_of(Role(role))
        return card.model_on_card(picked.engine, picked.name)
    except StandFault:
        raise
    except Exception as e:
        log.warning("run_snapshot.placement_unread", role=role, error=str(e))
        return None


# the model a role of this run answers with, the arm's own generator before the role's
def model_of(role: Role, model: str | None = None) -> llm.Resolved:
    return llm.resolve_for(role, model) if role is Role.generation else llm.resolve(role)


# the comment below promises the report survives an unreadable registry, so this one does too
def _generator(model: str | None):
    try:
        return model_of(Role.generation, model)
    except Exception as e:
        log.warning("run_snapshot.generator_unread", model=model, error=str(e))
        return llm.Resolved(model or "?", None)


# by the generator's own engine: a vLLM generator recorded a null, since `/api/ps` is ollama's
def _window(picked) -> int | None:
    import engines

    if picked.engine is None:
        return None
    return engines.driver(picked.engine.kind).window(picked.engine, picked.name)


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
    placed_during=None,
    cross_encoder_used=None,
    generated: bool = True,
    **filled,
) -> dict:
    unknown = sorted(set(filled) - set(KEYS))
    if unknown:
        raise ValueError(f"the run snapshot has no place for {unknown}")
    picked = _generator(model)
    # the agent's gate can call the reranker without `use_rerank`, and the record names it then too
    reranked = use_rerank if cross_encoder_used is None else cross_encoder_used
    roles = (*ANSWERING, Role.reranking) if reranked else ANSWERING
    # a row answered with no generator call names no generator: the role's default was read as the arm's
    if not generated:
        roles = tuple(role for role in roles if role is not Role.generation)
    named, samplers, placed, added, cache_keys, parsers = _by_role(picked, roles)
    # read while the role worked: a phased run writes its rows after the embedder has left the card
    placed |= {role: on for role, on in llm.placed_in_calls().items() if on is not None}
    placed |= placed_during or {}
    common = {
        "schema": SCHEMA,
        "rerank": use_rerank,
        "rerank_device": (rerank_device or _rerank_device()) if reranked else None,
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
        "context_length": _window(picked) if generated else None,
        "engines": named,
        "engine_refused": {role: seen.dropped for role, seen in samplers.items()},
        "engine_added": added,
        "samplers": {role: seen.sent for role, seen in samplers.items()},
        "on_card": placed,
        "answer_parsers": parsers,
        "cache_keys": cache_keys,
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
