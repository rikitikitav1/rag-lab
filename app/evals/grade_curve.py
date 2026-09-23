"""The curve of a graded pass: what every cut keeps on one arm, and the cross-encoder on the same rows."""

import json
from pathlib import Path

import numpy as np
from evals import gold_classes, measurements, stats
from use_cases import grading

SCHEMA = 1
CUTS = (None, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
QUANTILES = tuple(round(0.05 * n, 2) for n in range(20))
ARMS = {"A": "rank", "B": "rerank_score"}

READS = (
    "one arm at a time, never the union: arm A is the top of the fusion, the serving path;"
    " a chunk is kept when the model said yes with a probability at or above the cut, so the"
    " score of a verdict is `p` for yes and `1 - p` for no, and an unreadable answer keeps the"
    " chunk the way the live grader keeps it; retention is read on the rows whose gold section"
    " reached the arm, the floor on the rows that have a stranger there, and every share is a"
    " row's own, resampled over rows"
)


# the stand's own reader, never a second one: a copy of it kept a `no` the serving node drops
def _said(verdict: dict) -> tuple[str | None, float | None]:
    return grading.read_verdict(verdict.get("text")), verdict.get("p")


# the dial the cut is read on: one number that orders a confident no below a hesitant yes
def score_of(verdict: dict) -> float | None:
    said, p = _said(verdict)
    if said not in ("yes", "no"):
        return None
    # both words where the record has them: under a grammar the rest of the mass is not the other word
    top = verdict.get("top") or {}
    if "yes" in top and "no" in top and (top["yes"] + top["no"]) > 0:
        return top["yes"] / (top["yes"] + top["no"])
    if p is None:
        return None
    return p if said == "yes" else 1.0 - p


def _verdicts(row: dict) -> dict:
    return {v["key"]: v for v in row.get("verdicts") or ()}


# no cut is the serving node itself: it keeps by the word, and a `no` said at 0.36 is still a `no`
def _kept(verdict: dict | None, cut: float | None) -> bool:
    if verdict is None:
        return True
    said, _ = _said(verdict)
    if said not in ("yes", "no"):
        return True
    if cut is None:
        return said == "yes"
    score = score_of(verdict)
    return score is None or score >= cut


def kept_by_grader(row: dict, cut: float | None, form: str) -> set:
    verdicts = _verdicts(row)
    if form == "whole_text":
        kept = _kept(next(iter(verdicts.values()), None), cut)
        return set(range(len(row["addresses"]))) if kept else set()
    return {nth for nth, address in enumerate(row["addresses"])
            if _kept(verdicts.get(address), cut)}


def arm_of(frozen_row: dict, top: int, name: str) -> set:
    key = ARMS[name]
    ordered = sorted(
        frozen_row["candidates"],
        key=(lambda c: c["rank"]) if key == "rank" else (lambda c: -(c.get("rerank_score") or 0)),
    )
    return {c["address"] for c in ordered[:top]}


def shares(classes: list, addresses: list, arm: set, kept: set) -> dict:
    seats = [n for n, a in enumerate(addresses) if a in arm]
    of = lambda name: [n for n in seats if classes[n] == name]  # noqa: E731
    gold, neighbours, strangers = (of(c) for c in
                                   (gold_classes.GOLD, gold_classes.NEIGHBOUR, gold_classes.STRANGER))
    dropped = lambda seen: sum(1 for n in seen if n not in kept) / len(seen)  # noqa: E731
    return {
        "gold_any": float(any(n in kept for n in gold)) if gold else None,
        "gold_share": 1 - dropped(gold) if gold else None,
        "neighbours_dropped": dropped(neighbours) if neighbours else None,
        "strangers_dropped": dropped(strangers) if strangers else None,
    }


def _column(values: list, seed: int) -> dict:
    kept = [v for v in values if v is not None]
    if not kept:
        return {"point": None, "low": None, "n": 0}
    low, high = stats.bootstrap_ci(kept, seed=seed)
    return {"point": round(float(np.mean(kept)), 4), "low": round(low, 4),
            "high": round(high, 4), "n": len(kept)}


def _point(rows: list, seed: int) -> dict:
    return {name: _column([r[name] for r in rows], seed)
            for name in ("gold_any", "gold_share", "neighbours_dropped", "strangers_dropped")}


def _paired(rows_a: list, rows_b: list, name: str, seed: int) -> dict:
    deltas = [a[name] - b[name] for a, b in zip(rows_a, rows_b, strict=True)
              if a[name] is not None and b[name] is not None]
    if not deltas:
        return {"n": 0}
    low, high = stats.bootstrap_ci(deltas, seed=seed)
    return {"mean_delta": round(float(np.mean(deltas)), 4), "ci95": [round(low, 4), round(high, 4)],
            "n": len(deltas)}


def _thresholds(frozen: dict, rows: list, top: int, arm: str) -> list:
    seen = []
    for row in rows:
        source = frozen[row["question_id"]]
        keep = arm_of(source, top, arm)
        seen += [c.get("rerank_score") for c in source["candidates"]
                 if c["address"] in keep and c.get("rerank_score") is not None]
    if not seen:
        return []
    return [round(float(np.quantile(seen, q)), 4) for q in QUANTILES]


def _by_score(row: dict, frozen_row: dict, threshold: float) -> set:
    scored = {c["address"]: c.get("rerank_score") for c in frozen_row["candidates"]}
    return {n for n, a in enumerate(row["addresses"])
            if scored.get(a) is None or scored[a] >= threshold}


def curve(measurement: dict, frozen: dict, arm: str = "A", seed: int = 42) -> dict:
    top, form = measurement["top"], measurement["form"]
    by_id = {r["id"]: r for r in frozen["rows"]}
    rows = [r for r in measurement["rows"] if r["question_id"] in by_id]
    arms = {r["question_id"]: arm_of(by_id[r["question_id"]], top, arm) for r in rows}

    grader = []
    for cut in CUTS:
        read = [shares(r["classes"], r["addresses"], arms[r["question_id"]],
                       kept_by_grader(r, cut, form)) for r in rows]
        grader.append({"cut": cut, "rows": read, **_point(read, seed)})

    reranker = []
    for threshold in _thresholds(by_id, rows, top, arm):
        read = [shares(r["classes"], r["addresses"], arms[r["question_id"]],
                       _by_score(r, by_id[r["question_id"]], threshold)) for r in rows]
        reranker.append({"threshold": threshold, "rows": read, **_point(read, seed)})

    matched = []
    for step in grader:
        if step["gold_any"]["point"] is None or not reranker:
            continue
        near = min(reranker, key=lambda s: abs((s["gold_any"]["point"] or 0)
                                               - step["gold_any"]["point"]))
        matched.append({
            "cut": step["cut"],
            "threshold": near["threshold"],
            "retention": [step["gold_any"]["point"], near["gold_any"]["point"]],
            "strangers_dropped": [step["strangers_dropped"]["point"],
                                  near["strangers_dropped"]["point"]],
            "paired_delta": _paired(step["rows"], near["rows"], "strangers_dropped", seed),
        })

    strip = lambda steps: [{k: v for k, v in s.items() if k != "rows"} for s in steps]  # noqa: E731
    return {
        "schema": SCHEMA,
        "arm": arm,
        "form": form,
        "top": top,
        "questions": len(rows),
        "grader": strip(grader),
        "reranker": strip(reranker),
        "matched_retention": matched,
        "reads": READS,
    }


def run(measurement_path: str, candidates_path: str, arm: str = "A",
        name: str | None = None, record: bool = False) -> dict:
    measurement = json.loads(Path(measurement_path).read_text())
    measurement["rows"] = measurements.rows_of(measurement_path)
    frozen = json.loads(Path(candidates_path).read_text())
    payload = curve(measurement, frozen, arm)
    payload["measurement_file"] = Path(measurement_path).name
    payload["candidates_file"] = Path(candidates_path).name
    payload["candidates_stamp"] = frozen["stamp"]
    payload["where"] = measurements.say_where(
        "grade_curve", name or f"{Path(measurement_path).stem}_arm{arm}", payload, record
    )
    return payload
