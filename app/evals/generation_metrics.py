import sys
from collections import Counter

from evals.loaders import load_logs


def evaluate(run_name=None, verbose=False) -> dict:
    logs = load_logs(run_name)
    in_corpus = [ql for ql in logs if ql.question and ql.question.marked_sources]
    out_of_corpus = [ql for ql in logs if not (ql.question and ql.question.marked_sources)]

    faith = Counter()
    relevance = Counter()
    completeness = Counter(ql.completeness for ql in logs if ql.completeness)
    for ql in in_corpus:
        if ql.faithfulness:
            faith[ql.faithfulness] += 1
        if ql.relevance:
            relevance[ql.relevance] += 1
        if verbose:
            print(
                f"Q: {ql.question.original_text}\n"
                f"  answer: {(ql.answer or '')[:90]}\n"
                f"  faith: {ql.faithfulness} | relevance: {ql.relevance} | complete: {ql.completeness}\n"
                f"  reasons: {ql.metrics.get('faithfulness')} | {ql.metrics.get('relevance')} | {ql.metrics.get('completeness')}\n"
            )

    correct = sum(
        1 for ql in out_of_corpus if ql.faithfulness == "faithful" or not ql.answered
    )

    return {
        "faithfulness": dict(faith),
        "relevance": dict(relevance),
        "completeness": dict(completeness),
        "refusal_accuracy": f"{correct}/{len(out_of_corpus)}",
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    run_name = args[0] if args else None
    print(evaluate(run_name, verbose="--verbose" in sys.argv))
