import sys

import config
from evals import guest_axes
from evals.loaders import load_logs
from evals.pools import ALL_OUTCOMES, SAID_NOTHING, answered_in_target, settled, split
from evals.pools import has_remote_evidence as _has_remote_evidence
from evals.pools import kind as _kind
from evals.pools import outcome as _outcome
from evals.stats import mean_of, score_of
from outcomes import Outcome
from use_cases import rejudge

# an answer standing on nothing the corpus gave it, whichever way it got there
_UNSUPPORTED = (
    Outcome.unsupported_answer, Outcome.answered_ungrounded, Outcome.narrated_call,
)


# a mean cannot tell sharper from kinder: v3 rose on both while its tens rose by half
def _distribution(scores) -> dict:
    vals = [s for s in (score_of(x) for x in scores) if s is not None]
    if not vals:
        return {"n": 0, "tens": None, "at_least_8": None}
    return {
        "n": len(vals),
        "tens": round(sum(1 for v in vals if v == 10) / len(vals), 4),
        "at_least_8": round(sum(1 for v in vals if v >= 8) / len(vals), 4),
    }


def _scored(ql) -> bool:
    return any(getattr(ql, axis) is not None for axis in rejudge.AXES)


# read off the rule rather than restated beside it: two spellings of one table is the usual defect
def _abstentions() -> dict:
    return {
        "ours": {
            "outcomes": ["refused", "unsupported_answer"],
            "axes": list(rejudge.AXES),
            "why": "on a refusal the axis does not apply; the judge scores answers that cite the corpus, and one "
                   "without sources is counted in the unsupported shares instead",
            "read_from": "metrics.refusal and the row's answered flag, both written by the answering paths",
        },
        # read off the guests themselves: a fourth axis was added and this table did not notice
        "guests": {
            axis: "abstains where the row carries no " + ", no ".join(guest.needs)
            for axis, guest in guest_axes.AXES.items()
        },
    }


def _language_match(logs) -> dict:
    checked = [got for got in (answered_in_target(ql) for ql in logs) if got is not None]
    matched = sum(1 for got in checked if got)
    return {
        "n": len(checked),
        "matched": matched,
        "share": round(matched / len(checked), 3) if checked else None,
        "target": "the run's recorded language, or the question's where the run recorded none",
        "population": "rows that answered: refusals, narrated calls, errors and exhaustion are out",
        # the detector reads a config mode, so a run and a rerun can disagree without the code moving
        "detector": config.settings.retrieval.keyword.query_lang,
    }


# the guests had no door of their own: a fourth axis was scored and its mean lived in nobody's report
def _guests(logs) -> dict:
    out = {}
    for axis in guest_axes.AXES:
        seen = [(ql.metrics or {}).get(axis) or {} for ql in logs]
        scored = [one["score"] for one in seen if one.get("score") is not None]
        out[axis] = {
            "n": len(scored),
            "mean": mean_of(scored, 4),
            "abstained": sum(1 for one in seen if one.get("abstained")),
        }
    return out


def _share(logs, outcome) -> str:
    return f"{sum(1 for ql in logs if _outcome(ql) == outcome)}/{len(logs)}"


# 1 before `answered_ungrounded`; 2 those; 3 abstention; 4 settled; 5 language; 6 narrower; 7 guests
SCHEMA = 7


def evaluate(run_name=None, verbose=False) -> dict:
    logs = [ql for ql in load_logs(run_name) if _kind(ql) != "rejected"]
    # the same split `evals/pools` decides: this was three comprehensions repeating the rule
    pools = split(logs)
    in_corpus, off_domain, out_of_corpus = (
        pools["in_corpus"], pools["off_domain"], pools["out_of_corpus"]
    )

    # an ungrounded answer is still an answer, and its low scores belong here; one without sources has none
    answered_only = [
        ql for ql in in_corpus
        if _outcome(ql) not in SAID_NOTHING
    ]
    faith = mean_of(ql.faithfulness for ql in in_corpus)
    relevance = mean_of(ql.relevance for ql in in_corpus)
    completeness = mean_of(ql.completeness for ql in logs)

    if verbose:
        for ql in in_corpus:
            print(
                f"Q: {ql.question.original_text}\n"
                f"  answer: {(ql.answer or '')[:90]}\n"
                f"  faith: {ql.faithfulness} | relevance: {ql.relevance} | complete: {ql.completeness}\n"
            )

    via_remote = [ql for ql in out_of_corpus if _has_remote_evidence(ql)]
    refusal_pool = [ql for ql in out_of_corpus if not _has_remote_evidence(ql)]
    correct = sum(1 for ql in refusal_pool if _outcome(ql) == Outcome.refused)
    n = sum(1 for ql in in_corpus if score_of(ql.faithfulness) is not None)

    def norm(x):
        return round(x / 10, 3) if x is not None else None

    return {
        "schema": SCHEMA,
        # what the silence in an axis means: an abstention is not a low score and not a missing pass
        "axes_abstain_on": _abstentions(),
        "n_logs": len(logs),
        # the judge overrode what the answer wrote: nothing else is settled, the rest still derives
        "outcomes_overridden_by_the_judge": sum(1 for ql in logs if settled(ql)),
        # no model call and no judge: the same rule that picks the search config reads the answer
        "language_match": _language_match(logs),
        # a calibration, not an axis: reported beside ours, never blended into them
        "guest_axes": _guests(logs),
        "n_scored": n,
        "answered": sum(1 for ql in logs if ql.answered),
        "answer_rate": round(sum(1 for ql in logs if ql.answered) / len(logs), 3) if logs else None,
        "outcomes": {
            o: sum(1 for ql in logs if _outcome(ql) == o)
            for o in ALL_OUTCOMES
        },
        "faithfulness": faith,
        "relevance": relevance,
        "completeness": completeness,
        "distribution": {
            "faithfulness": _distribution(ql.faithfulness for ql in in_corpus),
            "relevance": _distribution(ql.relevance for ql in in_corpus),
            "completeness": _distribution(ql.completeness for ql in logs),
        },
        # a refusal takes a ten and a zero by the prompts, so its share moves both means
        "answered_only": {
            # the rows a mean stands on, not the rows with text: an unsupported answer carries no score
            "n": sum(1 for ql in answered_only if _scored(ql)),
            "without_a_score": sum(1 for ql in answered_only if not _scored(ql)),
            "faithfulness": mean_of(ql.faithfulness for ql in answered_only),
            "relevance": mean_of(ql.relevance for ql in answered_only),
            "completeness": mean_of(ql.completeness for ql in answered_only),
        },
        "faithfulness_0_1": norm(faith),
        "relevance_0_1": norm(relevance),
        "completeness_0_1": norm(completeness),
        "remote_grounding": mean_of(ql.faithfulness for ql in via_remote),
        "remote_relevance": mean_of(ql.relevance for ql in via_remote),
        "n_remote_scored": sum(1 for ql in via_remote if score_of(ql.faithfulness) is not None),
        "refusal_accuracy": f"{correct}/{len(refusal_pool)}",
        "off_domain_refusal": _share(off_domain, Outcome.refused),
        "off_domain_via_remote": sum(1 for ql in off_domain if _has_remote_evidence(ql)),
        "false_refusal": _share(in_corpus, Outcome.refused),
        "unsupported_in_corpus": _share(in_corpus, Outcome.unsupported_answer),
        "unsupported_external": _share(out_of_corpus, Outcome.unsupported_answer),
        "unsupported_off_domain": _share(off_domain, Outcome.unsupported_answer),
        "narrated_calls": sum(1 for ql in logs if _outcome(ql) == Outcome.narrated_call),
        "off_domain_grounding": mean_of(ql.faithfulness for ql in off_domain),
        "off_domain_refusal_rate": (
            round(
                sum(1 for ql in off_domain if _outcome(ql) == Outcome.refused)
                / len(off_domain),
                3,
            )
            if off_domain
            else None
        ),
        "supported_rate": (
            round(
                sum(1 for ql in logs if _outcome(ql) not in _UNSUPPORTED)
                / len(logs),
                3,
            )
            if logs
            else None
        ),
        "n_off_domain_scored": sum(1 for ql in off_domain if score_of(ql.faithfulness) is not None),
        "refused_with_context": sum(
            1 for ql in logs if _outcome(ql) == Outcome.refused and ql.sources
        ),
        "in_corpus_via_remote": sum(1 for ql in in_corpus if _has_remote_evidence(ql)),
        "answered_via_remote": len(via_remote),
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    run_name = args[0] if args else None
    print(evaluate(run_name, verbose="--verbose" in sys.argv))
