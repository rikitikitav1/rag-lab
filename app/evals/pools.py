import outcomes
from evals.stats import score_of
from outcomes import Outcome

# the taxonomy from the enum: a fourth bucket appeared while the pre-registration had three
ALL_OUTCOMES = tuple(o.value for o in outcomes.Outcome)

POOLS = ("in_corpus", "out_of_corpus", "off_domain", "rejected")


# the rule lives here alone: the set inventory asks the same question of a question, not a log
def kind_of_question(question) -> str:
    declared = question.kind if question else None
    if declared in POOLS:
        return declared
    return "in_corpus" if (question.marked_sources if question else None) else "out_of_corpus"


def kind(ql) -> str:
    return kind_of_question(ql.question if ql else None)


# pinned, not live: max_hops has only ever been 4, so a row keeps the ceiling it ran under
_CEILING_BEFORE_ROWS_RECORDED_IT = 4


# the row says which edge ended it; the ceiling is re-derived only for rows written before it did
def _exhausted(metrics: dict, snapshot: dict) -> bool:
    from use_cases.agent_policy import FinishedBy

    said = metrics.get("finished_by")
    # `unrecorded` is the bare arm saying it has no edge of ours, so the ceiling is re-derived
    if said and said != FinishedBy.unrecorded:
        return said == FinishedBy.hops_exhausted and not metrics.get("failed")
    its_ceiling = snapshot.get("max_hops")
    ceiling = _CEILING_BEFORE_ROWS_RECORDED_IT if its_ceiling is None else its_ceiling
    return (
        metrics.get("hops") is not None
        and metrics["hops"] >= ceiling
        and not metrics.get("failed")
    )


def outcome(ql) -> str:
    metrics = ql.metrics or {}
    recorded = metrics.get("outcome")
    if recorded in (Outcome.narrated_call, Outcome.exhausted):
        return recorded
    snapshot = metrics.get("config") or {}
    exhausted = _exhausted(metrics, snapshot)
    if recorded == Outcome.error:
        return Outcome.exhausted if exhausted else Outcome.error
    return outcomes.classify(
        ql.answer,
        bool(ql.sources),
        prefixes=[f"{name}__" for name in snapshot.get("mcp_configured") or []],
        exhausted=exhausted,
        grounded=None if ql.faithfulness is None else score_of(ql.faithfulness) > 0,
    )


# named once so a report and a correlation cannot narrow differently and be read side by side
IN_CORPUS_AND_ANSWERED = "the corpus pool, answered: marked sources, and our own outcome `answered`"


def in_corpus(ql) -> bool:
    return bool(ql.question and ql.question.marked_sources)


def in_corpus_and_answered(ql) -> bool:
    return in_corpus(ql) and outcome(ql) == Outcome.answered


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
