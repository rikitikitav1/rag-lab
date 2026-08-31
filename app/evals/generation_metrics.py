import sys

from evals.loaders import load_logs
from evals.pools import ALL_OUTCOMES
from evals.pools import has_remote_evidence as _has_remote_evidence
from evals.pools import kind as _kind
from evals.pools import outcome as _outcome


def _num(v):
    return None if v is None else int(v)


def _avg(scores):
    vals = [s for s in (_num(x) for x in scores) if s is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _share(logs, outcome) -> str:
    return f"{sum(1 for ql in logs if _outcome(ql) == outcome)}/{len(logs)}"


def evaluate(run_name=None, verbose=False) -> dict:
    logs = [ql for ql in load_logs(run_name) if _kind(ql) != "rejected"]
    in_corpus = [ql for ql in logs if _kind(ql) == "in_corpus" and ql.question.marked_sources]
    off_domain = [ql for ql in logs if _kind(ql) == "off_domain"]
    out_of_corpus = [
        ql for ql in logs if _kind(ql) == "out_of_corpus" or (
            _kind(ql) == "in_corpus" and not ql.question.marked_sources
        )
    ]

    faith = _avg(ql.faithfulness for ql in in_corpus)
    relevance = _avg(ql.relevance for ql in in_corpus)
    completeness = _avg(ql.completeness for ql in logs)

    if verbose:
        for ql in in_corpus:
            print(
                f"Q: {ql.question.original_text}\n"
                f"  answer: {(ql.answer or '')[:90]}\n"
                f"  faith: {ql.faithfulness} | relevance: {ql.relevance} | complete: {ql.completeness}\n"
            )

    via_remote = [ql for ql in out_of_corpus if _has_remote_evidence(ql)]
    refusal_pool = [ql for ql in out_of_corpus if not _has_remote_evidence(ql)]
    correct = sum(1 for ql in refusal_pool if _outcome(ql) == "refused")
    n = sum(1 for ql in in_corpus if _num(ql.faithfulness) is not None)

    def norm(x):
        return round(x / 10, 3) if x is not None else None

    return {
        "n_logs": len(logs),
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
        "faithfulness_0_1": norm(faith),
        "relevance_0_1": norm(relevance),
        "completeness_0_1": norm(completeness),
        "remote_grounding": _avg(ql.faithfulness for ql in via_remote),
        "remote_relevance": _avg(ql.relevance for ql in via_remote),
        "n_remote_scored": sum(1 for ql in via_remote if _num(ql.faithfulness) is not None),
        "refusal_accuracy": f"{correct}/{len(refusal_pool)}",
        "off_domain_refusal": _share(off_domain, "refused"),
        "off_domain_via_remote": sum(1 for ql in off_domain if _has_remote_evidence(ql)),
        "false_refusal": _share(in_corpus, "refused"),
        "unsupported_in_corpus": _share(in_corpus, "unsupported_answer"),
        "unsupported_external": _share(out_of_corpus, "unsupported_answer"),
        "unsupported_off_domain": _share(off_domain, "unsupported_answer"),
        "narrated_calls": sum(1 for ql in logs if _outcome(ql) == "narrated_call"),
        "off_domain_grounding": _avg(ql.faithfulness for ql in off_domain),
        "off_domain_refusal_rate": (
            round(sum(1 for ql in off_domain if _outcome(ql) == "refused") / len(off_domain), 3)
            if off_domain
            else None
        ),
        "supported_rate": (
            round(
                sum(
                    1 for ql in logs
                    if _outcome(ql) not in ("unsupported_answer", "narrated_call")
                )
                / len(logs),
                3,
            )
            if logs
            else None
        ),
        "n_off_domain_scored": sum(1 for ql in off_domain if _num(ql.faithfulness) is not None),
        "refused_with_context": sum(
            1 for ql in logs if _outcome(ql) == "refused" and ql.sources
        ),
        "in_corpus_via_remote": sum(1 for ql in in_corpus if _has_remote_evidence(ql)),
        "answered_via_remote": len(via_remote),
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    run_name = args[0] if args else None
    print(evaluate(run_name, verbose="--verbose" in sys.argv))
