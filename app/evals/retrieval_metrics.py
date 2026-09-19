import sys

import logging_setup
from evals.loaders import load_logs
from models.registry import Pipeline
from sqlalchemy.exc import SQLAlchemyError

log = logging_setup.get_logger(__name__)


# a mark is a path fragment, so a source matches by containment: hit@k stands on this
def is_gold(source: str, marked) -> bool:
    return any(m in source for m in marked)


def rank_of_gold(sources, marked) -> int | None:
    return next((i for i, s in enumerate(sources, 1) if is_gold(s, marked)), None)


# 6 scores a section only where the corpus has one; 5 added the axes; 4 added `file_precision`
SCHEMA = 7


# the pair the standard takes as an id, collapsed as `ranked_lists` does: one metric, one spelling
def section_ids(chunks) -> list[tuple[str, str | None]]:
    from use_cases.retrieval_compare import heading_text

    out = []
    for c in (chunks or []):
        if not c:
            continue
        key = (c["source"], heading_text(c.get("section")))
        if key not in out:
            out.append(key)
    return out


# the gold section by its own rank, so a chunk of the right file in the wrong section is not a hit
def rank_of_gold_section(chunks, marked, gold_heading) -> int | None:
    from use_cases.retrieval_compare import rank_of_section

    return rank_of_section(section_ids(chunks), marked, gold_heading)


# what share of the files retrieval reached were gold; the same population `hit@k` ranks
def file_precision(got: list[str], marked) -> float | None:
    shown = {next((m for m in marked if m in s), s) for s in got}
    return len(shown & set(marked)) / len(shown) if shown else None


# what a row is ranked on, in the order it is ranked: the second reader of this had to guess
def retrieved_sources(ql) -> tuple[list[dict], list[str], list[str]]:
    kept = [s for s in (ql.sources or []) if not s["source"].startswith("mcp:")]
    # a policy that hides weak chunks from the model still retrieved them
    dropped = ((ql.metrics or {}).get("retrieval") or {}).get("dropped_sources") or []
    return kept, dropped, [s["source"] for s in kept] + dropped


# the source question's own text, a join away; it resolves everywhere but hand-written `curated`
def _gold_headings(logs) -> dict[int, str]:
    from models.eval import Question
    from orm.sync_db import Session
    from sqlalchemy import select

    ids = {ql.question_id for ql in logs if ql.question_id}
    if not ids:
        return {}
    source = {ql.question_id: ql.question.source_question_id for ql in logs if ql.question_id}
    wanted = {v for v in source.values() if v}
    with Session() as session:
        texts = dict(
            session.execute(
                select(Question.id, Question.original_text).where(Question.id.in_(wanted))
            ).all()
        ) if wanted else {}
    # the same COALESCE `retrieval_compare.questions` writes in sql: two doors, one population
    by_id = {ql.question_id: ql.question.original_text for ql in logs if ql.question_id}
    return {qid: texts.get(src) or by_id.get(qid) for qid, src in source.items()}


# scorable where the corpus has that section, keyed by the python object and not by a column
def _scorable_sections(in_corpus, golds) -> set[int]:
    from orm.sync_db import engine
    from use_cases.retrieval_compare import section_exists

    wanted = [ql for ql in in_corpus if ql.chunks and golds.get(ql.question_id)]
    if not wanted:
        return set()
    try:
        with engine.connect() as conn:
            return {
                id(ql)
                for ql in wanted
                if section_exists(
                    conn,
                    ((ql.metrics or {}).get("config") or {}).get("variant"),
                    ql.question.marked_sources,
                    golds[ql.question_id],
                )
            }
    except SQLAlchemyError as e:
        # the axis is worth less than the report: a probe that cannot run scores nothing
        log.warning("retrieval_metrics.sections_unknown", error=str(e))
        return set()


def evaluate(run_name=None):
    logs = load_logs(run_name)
    in_corpus = [ql for ql in logs if ql.question and ql.question.marked_sources]

    hits, rr_sum, misses = 0, 0.0, []
    rr_in_hop, found_at_hop, hop_unknown, in_hop_n = 0.0, {}, 0, 0
    precisions: list[float] = []
    section_hits, section_rr, section_scored = 0, 0.0, 0
    golds = _gold_headings(in_corpus)
    scorable = _scorable_sections(in_corpus, golds)
    for ql in in_corpus:
        expected = ql.question.marked_sources
        # the section axes see what the gate left; the file axes see what search found, gate aside
        gold_heading = golds.get(ql.question_id)
        if id(ql) in scorable:
            section_scored += 1
            section_rank = rank_of_gold_section(ql.chunks, expected, gold_heading)
            if section_rank:
                section_hits += 1
                section_rr += 1 / section_rank
        kept, dropped, got = retrieved_sources(ql)
        precision = file_precision(got, expected)
        if precision is not None:
            precisions.append(precision)
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

    # no corpus row is nothing measured, not a measured zero: the other axes already read it so
    n = len(in_corpus)
    return {
        "schema": SCHEMA,
        "hit_at_k": round(hits / n, 3) if n else None,
        "mrr": round(rr_sum / n, 3) if n else None,
        "hits": hits,
        "n": len(in_corpus),
        "misses": len(misses),
        # gated-away files count here as they do in the rank: retrieval reached them
        "file_precision": round(sum(precisions) / len(precisions), 3) if precisions else None,
        "n_precision_scored": len(precisions),
        # a chunk of the right file in the wrong section is a hit above and a miss here
        "section_hit_at_k": round(section_hits / section_scored, 3) if section_scored else None,
        "section_mrr": round(section_rr / section_scored, 3) if section_scored else None,
        "n_section_scored": section_scored,
        # the two pairs differ by grain and by gate, and only the grain is in their names
        "section_axes_see_the_kept_chunks": True,
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
