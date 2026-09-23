import outcomes
from evals.stats import score_of
from outcomes import Outcome

import db

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


# an override, not a key: 266 rows carry a settlement equal to what the answer already knew
def _override(metrics: dict) -> str | None:
    got = metrics.get("settled_outcome")
    return got if got and got != metrics.get("outcome") else None


# the hop ceiling read the way the outcome reads it, for a column that names the edge alone
def hops_exhausted(ql) -> bool:
    metrics = ql.metrics or {}
    return _exhausted(metrics, metrics.get("config") or {})


def settled(ql) -> bool:
    return _override(ql.metrics or {}) is not None


def outcome(ql) -> str:
    metrics = ql.metrics or {}
    # only the judge can settle groundedness, so only its key is trusted whole
    override = _override(metrics)
    if override:
        return override
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


# narrower than any pool, and named apart: one label used to stand over two selections
JOINS_BOTH_JUDGES = (
    "the corpus pool, answered, and carrying all three of our faithfulness, a context and the"
    " guest's faithfulness, since a correlation needs both scores on one row"
)


def guest_score(ql, axis: str):
    return ((ql.metrics or {}).get(axis) or {}).get("score")


def joins_both_judges(ql) -> bool:
    return (
        in_corpus_and_answered(ql)
        and score_of(ql.faithfulness) is not None
        and bool(ql.contexts)
        and guest_score(ql, "ragas_faithfulness") is not None
    )


# refusals and non-answers: shapes where the model said nothing to score
SAID_NOTHING = (
    Outcome.refused, Outcome.narrated_call, Outcome.exhausted, Outcome.error,
)


# the run's own language when it recorded one, else the question's: answering the asker is the default
def target_language(ql) -> str | None:
    asked = ((ql.metrics or {}).get("config") or {}).get("language")
    return db.resolve_language(ql.question_text, asked) if ql.question_text else asked


# None where the question cannot be put: a narrated tool call is json, not an answer in a language
def answered_in_target(ql) -> bool | None:
    target = target_language(ql)
    if not ql.answer or not target or outcome(ql) in SAID_NOTHING:
        return None
    return db.detect_language(ql.answer) == target


# four doors paired by question with three rules: two kept the last row, one refused, one made a set
class Ambiguous(ValueError):
    pass


def by_question(logs, keep=None) -> dict:
    kept = {}
    for ql in logs:
        if ql.question_id is None or (keep and not keep(ql)):
            continue
        if ql.question_id in kept:
            raise Ambiguous(
                f"question {ql.question_id} appears twice in run {ql.run_name!r}: whichever row"
                " a pairing kept would depend on the order the loader returned them"
            )
        kept[ql.question_id] = ql
    return kept
