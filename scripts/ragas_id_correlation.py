"""Our ranks against the standard's judge-free half: ID metrics over the same rows.

Reads the archive, scores nothing with a model, touches no card. One of the three comparisons
is an identity by construction and is declared as such in the checklist before this ran.
"""

import argparse
import asyncio
import json
from pathlib import Path

from evals.loaders import load_logs
from evals.retrieval_metrics import (
    _gold_headings,
    is_gold,
    rank_of_gold,
    rank_of_gold_section,
    retrieved_sources,
    section_ids,
)
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent

# 1 this report's own version: it used to sign itself with `retrieval_metrics.SCHEMA`
SCHEMA = 1


# ragas compares ids by equality, `is_gold` by containment: reduce before handing them over
def as_ids(sources: list[str], marked: list[str]) -> list[str]:
    return [next((m for m in marked if m in s), s) for s in sources]


# the address of a chunk, in the shape ragas takes as an id
def as_section_ids(chunks, marked, gold_heading) -> tuple[list[str], list[str]]:
    from use_cases.retrieval_compare import clean_gold

    # a headless chunk keeps its place, or `rr` and `id_precision` count different populations
    retrieved = [
        f"{next((m for m in marked if m in src), src)}#{head or ''}"
        for src, head in section_ids(chunks)
    ]
    # every marked source, not the first: the file level hands `list(marked)` over whole
    return retrieved, [f"{m}#{clean_gold(gold_heading)}" for m in marked]


def rows_of(run_name=None, level="file") -> list[dict]:
    out = []
    golds = _gold_headings(load_logs(run_name)) if level == "section" else {}
    for ql in load_logs(run_name):
        marked = ql.question.marked_sources if ql.question else None
        if not marked:
            continue
        if level == "section":
            gold = golds.get(ql.question_id)
            if not (ql.chunks and gold):
                continue
            retrieved, reference = as_section_ids(ql.chunks, marked, gold)
            if not retrieved:
                continue
            rank = rank_of_gold_section(ql.chunks, marked, gold)
            out.append({
                "id": ql.id, "run_name": ql.run_name, "pipeline": str(ql.pipeline),
                "retrieved": retrieved, "reference": reference,
                "hit": 1.0 if rank else 0.0, "rr": 1 / rank if rank else 0.0,
                "n_retrieved": len(retrieved),
                "n_gold_retrieved": sum(1 for r in retrieved if r in reference),
            })
            continue
        _kept, _dropped, got = retrieved_sources(ql)
        if not got:
            continue
        rank = rank_of_gold(got, marked)
        out.append({
            "id": ql.id,
            "run_name": ql.run_name,
            "pipeline": str(ql.pipeline),
            "retrieved": as_ids(got, marked),
            "reference": list(marked),
            "hit": 1.0 if rank else 0.0,
            "rr": 1 / rank if rank else 0.0,
            "n_retrieved": len(got),
            "n_gold_retrieved": sum(1 for s in got if is_gold(s, marked)),
        })
    return out


def scored(rows: list[dict]) -> list[dict]:
    recall, precision = IDBasedContextRecall(), IDBasedContextPrecision()

    async def both(row):
        sample = SingleTurnSample(
            retrieved_context_ids=row["retrieved"], reference_context_ids=row["reference"]
        )
        return (await recall.single_turn_ascore(sample),
                await precision.single_turn_ascore(sample))

    async def all_rows():
        return [await both(r) for r in rows]

    for row, (rec, prec) in zip(rows, asyncio.run(all_rows()), strict=True):
        row["id_recall"], row["id_precision"] = rec, prec
    return rows


def correlate(rows: list[dict], left: str, right: str) -> dict:
    a = [r[left] for r in rows]
    b = [r[right] for r in rows]
    # a column that never varies has no rank order, and scipy answers nan rather than refusing
    if len(set(a)) < 2 or len(set(b)) < 2:
        return {"rho": None, "p": None, "n": len(rows), "reason": "a column does not vary"}
    result = spearmanr(a, b)
    return {"rho": round(float(result.statistic), 4), "p": round(float(result.pvalue), 6),
            "n": len(rows)}


def report(rows: list[dict], level: str = "file") -> dict:
    one_label = [r for r in rows if len(r["reference"]) == 1]
    headless = sum(1 for r in rows if any(i.endswith("#") for i in r["retrieved"]))
    return {
        "schema": SCHEMA,
        "level": level,
        "n": len(rows),
        "n_one_label": len(one_label),
        "n_rows_with_a_headless_chunk": headless,
        "runs": len({r["run_name"] for r in rows}),
        "pipelines": sorted({r["pipeline"] for r in rows}),
        "means": {
            key: round(sum(r[key] for r in rows) / len(rows), 4)
            for key in ("hit", "rr", "id_recall", "id_precision")
        },
        "declared_identity": {
            "claim": "with one label per question ID-recall is hit@k row by row",
            "rows_where_they_differ": sum(1 for r in one_label if r["hit"] != r["id_recall"]),
            "checked_on": len(one_label),
            # measured, not assumed: at section level the reference is one id by construction
            "filter_selects": len(one_label) < len(rows),
        },
        "correlations": {
            "hit_vs_id_recall": correlate(rows, "hit", "id_recall"),
            "mrr_vs_id_recall": correlate(rows, "rr", "id_recall"),
            "mrr_vs_id_precision": correlate(rows, "rr", "id_precision"),
            "hit_vs_id_precision": correlate(rows, "hit", "id_precision"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None, help="one run; default is every row")
    parser.add_argument("--out", default=None, help="write the report here as json")
    parser.add_argument("--level", choices=("file", "section"), default="file",
                        help="what an id is: the file, or the file and its section")
    args = parser.parse_args()

    rows = scored(rows_of(args.run_name, args.level))
    if not rows:
        raise SystemExit("no rows carry both a marked source and retrieved sources")
    out = report(rows, args.level)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
