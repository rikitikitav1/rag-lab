import sys

from evals.loaders import load_logs
from evals.pools import ALL_OUTCOMES, settled, split
from evals.pools import has_remote_evidence as _has_remote_evidence
from evals.pools import kind as _kind
from evals.pools import outcome as _outcome
from evals.stats import mean_of, score_of
from outcomes import Outcome
from use_cases import rejudge

# refusals and non-answers: shapes where the model said nothing to score
_SAID_NOTHING = (
    Outcome.refused, Outcome.narrated_call, Outcome.exhausted, Outcome.error,
)
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


# read off the rule rather than restated beside it: two spellings of one table is the usual defect
def _abstentions() -> dict:
    return {
        "ours": {
            "outcomes": ["refused"],
            "axes": list(rejudge.AXES),
            "why": "on a refusal the axis does not apply, and the judge prompt is left alone",
            "read_from": "metrics.refusal, written by both answering paths from one function",
        },
        "guests": {
            "ragas_faithfulness": "abstains where the row carries no answer or no context",
            "ragas_context_precision": "abstains where the question carries no reference answer",
            "ragas_context_recall": "abstains where the question carries no reference answer",
        },
    }


def _share(logs, outcome) -> str:
    return f"{sum(1 for ql in logs if _outcome(ql) == outcome)}/{len(logs)}"


# 1 before `answered_ungrounded`; 2 adds those three; 3 where the axes abstain; 4 who settled
SCHEMA = 4


def evaluate(run_name=None, verbose=False) -> dict:
    logs = [ql for ql in load_logs(run_name) if _kind(ql) != "rejected"]
    # the same split `evals/pools` decides: this was three comprehensions repeating the rule
    pools = split(logs)
    in_corpus, off_domain, out_of_corpus = (
        pools["in_corpus"], pools["off_domain"], pools["out_of_corpus"]
    )

    # an ungrounded answer is still an answer, and its low scores belong in this mean
    answered_only = [
        ql for ql in in_corpus
        if _outcome(ql) not in _SAID_NOTHING
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
        # a derived outcome is a guess about groundedness, and it must not read as a recorded one
        "outcomes_settled": sum(1 for ql in logs if settled(ql)),
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
            "n": len(answered_only),
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
