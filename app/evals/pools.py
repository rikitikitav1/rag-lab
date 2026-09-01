import config
import outcomes
from evals.stats import score_of
from outcomes import Outcome

# the taxonomy from the enum: a fourth bucket appeared while the pre-registration had three
ALL_OUTCOMES = tuple(o.value for o in outcomes.Outcome)

POOLS = ("in_corpus", "out_of_corpus", "off_domain", "rejected")


def kind(ql) -> str:
    marked = ql.question.marked_sources if ql.question else None
    declared = ql.question.kind if ql.question else None
    if declared in POOLS:
        return declared
    return "in_corpus" if marked else "out_of_corpus"


def outcome(ql) -> str:
    metrics = ql.metrics or {}
    recorded = metrics.get("outcome")
    if recorded in (Outcome.narrated_call, Outcome.exhausted):
        return recorded
    snapshot = metrics.get("config") or {}
    # the row's own ceiling, today's default only where it recorded none
    its_ceiling = snapshot.get("max_hops")
    ceiling = config.settings.agent.max_hops if its_ceiling is None else its_ceiling
    exhausted = (
        metrics.get("hops") is not None
        and metrics["hops"] >= ceiling
        and not metrics.get("failed")
    )
    if recorded == Outcome.error:
        return Outcome.exhausted if exhausted else Outcome.error
    return outcomes.classify(
        ql.answer,
        bool(ql.sources),
        prefixes=[f"{name}__" for name in snapshot.get("mcp_configured") or []],
        exhausted=exhausted,
        grounded=None if ql.faithfulness is None else score_of(ql.faithfulness) > 0,
    )


def has_remote_evidence(ql) -> bool:
    return any(s["source"].startswith("mcp:") for s in (ql.sources or []))


# in_corpus without marked sources cannot be scored against the corpus, so it lands outside
def split(logs) -> dict[str, list]:
    pools: dict[str, list] = {name: [] for name in POOLS}
    for ql in logs:
        name = kind(ql)
        if name == "in_corpus" and not (ql.question and ql.question.marked_sources):
            name = "out_of_corpus"
        pools[name].append(ql)
    return pools
