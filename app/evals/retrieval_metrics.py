import sys

from evals.loaders import load_logs


def evaluate(run_name=None):
    logs = load_logs(run_name)
    in_corpus = [ql for ql in logs if ql.question and ql.question.marked_sources]

    hits, rr_sum, misses = 0, 0.0, []
    for ql in in_corpus:
        expected = ql.question.marked_sources
        got = [
            s["source"]
            for s in (ql.sources or [])
            if not s["source"].startswith("mcp:")
        ]
        rank = None
        for i, source in enumerate(got, 1):
            if any(exp in source for exp in expected):
                rank = i
                break
        if rank:
            hits += 1
            rr_sum += 1 / rank
        else:
            misses.append(ql.question.original_text)

    n = len(in_corpus) or 1
    return {
        "hit_at_k": round(hits / n, 3),
        "mrr": round(rr_sum / n, 3),
        "hits": hits,
        "n": len(in_corpus),
        "misses": len(misses),
    }


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    r = evaluate(run_name)
    print(f"hit@k: {r['hits']}/{r['n']} = {r['hit_at_k']:.0%}")
    print(f"MRR:   {r['mrr']:.3f}")
    print("misses:", r["misses"])
