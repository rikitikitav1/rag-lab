import sys

from evals.loaders import load_logs


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _avg(scores):
    vals = [s for s in (_num(x) for x in scores) if s is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def evaluate(run_name=None, verbose=False) -> dict:
    logs = load_logs(run_name)
    in_corpus = [ql for ql in logs if ql.question and ql.question.marked_sources]
    out_of_corpus = [ql for ql in logs if not (ql.question and ql.question.marked_sources)]

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

    correct = sum(
        1 for ql in out_of_corpus if (_num(ql.faithfulness) or 0) >= 7 or not ql.answered
    )
    n = sum(1 for ql in in_corpus if _num(ql.faithfulness) is not None)

    def norm(x):
        return round(x / 10, 3) if x is not None else None

    return {
        "n_scored": n,
        "faithfulness": faith,
        "relevance": relevance,
        "completeness": completeness,
        "faithfulness_0_1": norm(faith),
        "relevance_0_1": norm(relevance),
        "completeness_0_1": norm(completeness),
        "refusal_accuracy": f"{correct}/{len(out_of_corpus)}",
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    run_name = args[0] if args else None
    print(evaluate(run_name, verbose="--verbose" in sys.argv))
