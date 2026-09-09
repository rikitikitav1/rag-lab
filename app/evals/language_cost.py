"""What our own axes charge when the answer comes back in the language it was asked in.

The number of 08.09 was computed from a shell, so nothing could recompute it, and the group it was
read on was carved out by the outcome of the change. Both cuts are here. The one to quote is
declared from the record as it stood before: rows where the baseline answered in another language
than the one asked. The other is the same reading on the group the outcome selected, kept beside it
because it is what was published, not because it is the better cut.
"""

import config
from evals.compare import SCHEMA as COMPARE_SCHEMA
from evals.compare import residencies
from evals.human_anchor import Refused
from evals.pools import answered_in_target, by_question, in_corpus_and_answered
from evals.stats import delta_stats, score_of, tally
from use_cases import rejudge

import db

# 1 both cuts, the floor, and the comparability of the two arms
SCHEMA = 1

POPULATION = (
    "the corpus pool, answered, present in both arms under one question_id and carrying the axis"
    " in both: one predicate, applied by this code"
)

# the cut that can be named without seeing the result, and the cut the result named
DECLARED_FROM_BEFORE = "answered_off_language_before"
SELECTED_BY_OUTCOME = "switched_language"

GROUPS = {
    DECLARED_FROM_BEFORE: "visible in the record before the change: quote this one",
    SELECTED_BY_OUTCOME: "selected by the outcome of the change: kept because it was published",
}


def _pairs(before_logs, after_logs) -> list[tuple]:
    before = by_question(before_logs, in_corpus_and_answered)
    after = by_question(after_logs, in_corpus_and_answered)
    return [(before[q], after[q]) for q in sorted(set(before) & set(after))]


def _in_group(name: str, before, after) -> bool:
    if name == DECLARED_FROM_BEFORE:
        return answered_in_target(before) is False
    return db.detect_language(before.answer or "") != db.detect_language(after.answer or "")


def _axis_deltas(pairs, axis) -> list[float]:
    scored = [
        (score_of(getattr(before, axis)), score_of(getattr(after, axis)))
        for before, after in pairs
    ]
    return [after - before for before, after in scored if before is not None and after is not None]


def _over(pairs) -> dict:
    out = {}
    for axis in rejudge.AXES:
        deltas = _axis_deltas(pairs, axis)
        if not deltas:
            out[axis] = {"n": 0, "unreadable": "no pair carries this axis on both sides"}
            continue
        out[axis] = {**delta_stats(deltas), **tally(deltas)}
    return out


def measure(before_run: str, after_run: str, floor_against: str | None = None) -> dict:
    from evals.loaders import load_logs

    if before_run == after_run:
        raise Refused("an arm against itself measures its own drift, and that is the floor block")
    before_logs, after_logs = load_logs(before_run), load_logs(after_run)
    pairs = _pairs(before_logs, after_logs)
    grouped = {
        name: [p for p in pairs if _in_group(name, *p)] for name in GROUPS
    }
    got = {
        "schema": SCHEMA,
        "arms": {"before": before_run, "after": after_run},
        "population": POPULATION,
        # the judge's scale is spelled two ways across records and no field said which
        "scale": "0..10, the judge's own",
        "detector": config.settings.retrieval.query_lang,
        "n_pairs": len(pairs),
        # two arms are one instrument only under one engine, one ruler and one residency
        "comparability": residencies({before_run: before_logs, after_run: after_logs}),
        # the block above is shaped by another report, so this record names that shape too
        "comparability_schema": COMPARE_SCHEMA,
        "cuts": {
            name: {
                "why": why,
                "n": len(grouped[name]),
                "axes": _over(grouped[name]),
            }
            for name, why in GROUPS.items()
        },
        "overlap_of_the_two_cuts": len(
            {before.question_id for before, _ in grouped[DECLARED_FROM_BEFORE]}
            & {before.question_id for before, _ in grouped[SELECTED_BY_OUTCOME]}
        ),
        "all_pairs": _over(pairs),
    }
    if floor_against:
        # the same answers judged across a reload: what this contrast cannot go below
        floor_logs = load_logs(floor_against)
        got["drift_floor"] = {
            "arms": {"before": floor_against, "after": before_run},
            "axes": _over(_pairs(floor_logs, before_logs)),
        }
    return got
