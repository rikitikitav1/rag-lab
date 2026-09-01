import sys

from evals.loaders import load_logs
from models.registry import Pipeline


# a mark is a path fragment, so a source matches by containment: hit@k stands on this
def is_gold(source: str, marked) -> bool:
    return any(m in source for m in marked)


def rank_of_gold(sources, marked) -> int | None:
    return next((i for i, s in enumerate(sources, 1) if is_gold(s, marked)), None)


# 3 stops putting an agent's dropped sources in hop 1, which is a hop they never recorded
SCHEMA = 3


def evaluate(run_name=None):
    logs = load_logs(run_name)
    in_corpus = [ql for ql in logs if ql.question and ql.question.marked_sources]

    hits, rr_sum, misses = 0, 0.0, []
    rr_in_hop, found_at_hop, hop_unknown, in_hop_n = 0.0, {}, 0, 0
    for ql in in_corpus:
        expected = ql.question.marked_sources
        kept = [s for s in (ql.sources or []) if not s["source"].startswith("mcp:")]
        got = [s["source"] for s in kept]
        # a policy that hides weak chunks from the model still retrieved them
        dropped = ((ql.metrics or {}).get("retrieval") or {}).get("dropped_sources") or []
        got += dropped
        rank = rank_of_gold(got, expected)
        if rank:
            hits += 1
            rr_sum += 1 / rank
            hop, hop_rank = (
                _rank_inside_its_hop(kept, dropped, expected, _one_hop_by_construction(ql))
                if _hops_are_recorded(ql, kept)
                else (None, None)
            )
            # the row cannot say where it was found, and "hop 1" would be a claim
            if hop_rank is None:
                hop_unknown += 1
            else:
                in_hop_n += 1
                rr_in_hop += 1 / hop_rank
                found_at_hop[hop] = found_at_hop.get(hop, 0) + 1
        else:
            misses.append(ql.question.original_text)

    n = len(in_corpus) or 1
    return {
        "schema": SCHEMA,
        "hit_at_k": round(hits / n, 3),
        "mrr": round(rr_sum / n, 3),
        "hits": hits,
        "n": len(in_corpus),
        "misses": len(misses),
        # a rank across a concatenation of retrievals is not a rank
        "mrr_in_hop": round(rr_in_hop / in_hop_n, 3) if in_hop_n else None,
        "found_at_hop": {str(k): v for k, v in sorted(found_at_hop.items())},
        "hop_unknown": hop_unknown,
    }


# an agent row written before the stamp is unknown, and "hop 1" would be a claim
def _hops_are_recorded(ql, kept: list) -> bool:
    return _one_hop_by_construction(ql) or all(
        s.get("hop") is not None for s in kept
    )


def _one_hop_by_construction(ql) -> bool:
    return getattr(ql, "pipeline", Pipeline.single_shot) != Pipeline.agent


# a single_shot row has one hop by construction, so `hop or 1` is the truth for it
def _rank_inside_its_hop(kept, dropped, expected, one_hop=True) -> tuple[int | None, int | None]:
    # an agent's dropped sources carry no hop, and calling them hop 1 is the refused claim
    at_hop_one = dropped if one_hop else []
    for hop in sorted({s.get("hop") or 1 for s in kept} | ({1} if at_hop_one else set())):
        ordered = [s["source"] for s in kept if (s.get("hop") or 1) == hop]
        if hop == 1:
            ordered += at_hop_one
        rank = rank_of_gold(ordered, expected)
        if rank:
            return hop, rank
    return None, None


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    r = evaluate(run_name)
    print(f"hit@k: {r['hits']}/{r['n']} = {r['hit_at_k']:.0%}")
    print(f"MRR:   {r['mrr']:.3f}")
    print("misses:", r["misses"])
