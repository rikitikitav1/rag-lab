"""Our judge against the standard's, on the same rows, under the predictions written before it ran.

The four predictions and what would refute each live in the arc log, section «Предрегистрация
корреляции». This module only measures them; it does not decide what they meant.
"""

import math
import re
import statistics

from evals.generation_metrics import _outcome, score_of
from evals.loaders import load_logs
from outcomes import Outcome
from scipy.stats import spearmanr
from use_cases.ingest_quality import code_fraction

# 3 rho carries its band; 2 was the corpus pool alone; 1 was every row that carried both scores
SCHEMA = 4

WORD = re.compile(r"\w+", re.U)

# the pre-registered thresholds, named before the count: moving one is a decision, not a tweak
RHO_WITH_OVERLAP = 0.3
PARTIAL_GAP = 0.1
STRATUM_GAP = 0.1
MIN_ROWS = 100
# what counts as a code-bearing context: `_is_code_only` fired on none of 729
CODE_STRATUM = 0.2


# the covariate: deterministic, no calls, a property of the answer rather than a cut through it
def overlap(answer: str, contexts: list[str], n: int = 4) -> float:
    def grams(text):
        words = WORD.findall((text or "").lower())
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    got = grams(answer)
    return round(len(got & grams("\n".join(contexts or []))) / len(got), 4) if got else 0.0


# how much of the context is code: `_is_code_only` was false for all 729 contexts of the first run
def code_share(contexts: list[str]) -> float:
    return statistics.fmean(code_fraction(c) for c in contexts) if contexts else 0.0


def guest_score(ql, axis: str):
    return ((ql.metrics or {}).get(axis) or {}).get("score")


def rows_of(run_name=None) -> tuple[list[dict], dict]:
    kept, refused, abstained, both, off_pool = [], 0, 0, 0, 0
    for ql in load_logs(run_name):
        ours = score_of(ql.faithfulness)
        guest = guest_score(ql, "ragas_faithfulness")
        entry = (ql.metrics or {}).get("ragas_faithfulness") or {}
        if entry.get("abstained"):
            abstained += 1
        if ours is None or not ql.contexts:
            continue
        # the corpus pool alone: three pools sit at three heights and their gap propped up rho
        if not (ql.question and ql.question.marked_sources):
            off_pool += 1
            continue
        # our own outcome decides, not the standard's `nan`: the owner's rule of 06.09
        if _outcome(ql) == Outcome.refused:
            refused += 1
            if entry.get("abstained"):
                both += 1
            continue
        # `answered` and `not refused` are not the same: a narrated call is neither
        if _outcome(ql) != Outcome.answered:
            off_pool += 1
            continue
        if guest is None:
            continue
        kept.append({
            # the arm copies carry new log ids, so only the question joins a row to its twin
            "id": ql.id, "question_id": getattr(ql, "question_id", None),
            "run_name": ql.run_name, "pipeline": str(ql.pipeline),
            "ours": ours / 10, "guest": guest,
            "overlap": overlap(ql.answer, ql.contexts),
            "code_share": code_share(ql.contexts),
            "on_card": entry.get("on_card"),
            "guest_precision": guest_score(ql, "ragas_context_precision"),
            "guest_recall": guest_score(ql, "ragas_context_recall"),
        })
    return kept, {"refused_excluded": refused, "guest_abstained": abstained,
                  "refused_and_abstained": both, "outside_the_declared_population": off_pool}


def rho(rows, left: str, right: str) -> float | None:
    a = [r[left] for r in rows]
    b = [r[right] for r in rows]
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return round(float(spearmanr(a, b).statistic), 4)


# spearman on ranks, so the partial is the same formula the pearson one uses
def partial(xy, xz, yz) -> float | None:
    if xy is None or xz is None or yz is None:
        return None
    bottom = math.sqrt((1 - xz**2) * (1 - yz**2))
    return round((xy - xz * yz) / bottom, 4) if bottom else None


def strata(rows) -> dict:
    coded = [r for r in rows if r["code_share"] >= CODE_STRATUM]
    prose = [r for r in rows if r["code_share"] < CODE_STRATUM]

    def mean(sample, key):
        return round(statistics.fmean(r[key] for r in sample), 4) if sample else None

    return {
        "code_contexts": {
            "n": len(coded), "guest": mean(coded, "guest"), "ours": mean(coded, "ours")
        },
        "prose_contexts": {
            "n": len(prose), "guest": mean(prose, "guest"), "ours": mean(prose, "ours")
        },
    }


# the pairs are resampled, not the coefficients: `bootstrap_ci` averages values and cannot do this
def rho_ci(rows, a: str, b: str, seed: int = 0) -> list | None:
    import random

    from use_cases.retrieval_compare import BOOTSTRAP

    if len(rows) < 3:
        return None
    rng, drawn = random.Random(seed), []
    for _ in range(BOOTSTRAP):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = rho(sample, a, b)
        if value is not None:
            drawn.append(value)
    if not drawn:
        return None
    drawn.sort()
    return [round(drawn[int(0.025 * len(drawn))], 4), round(drawn[int(0.975 * len(drawn))], 4)]


def report(rows, counts) -> dict:
    ours_guest = rho(rows, "ours", "guest")
    guest_overlap = rho(rows, "guest", "overlap")
    ours_overlap = rho(rows, "ours", "overlap")
    controlled = partial(ours_guest, ours_overlap, guest_overlap)
    split = strata(rows)
    coded, prose = split["code_contexts"]["guest"], split["prose_contexts"]["guest"]
    gap = abs(coded - prose) if coded is not None and prose is not None else None
    return {
        "schema": SCHEMA,
        "n": len(rows),
        "runs": sorted({r["run_name"] for r in rows}),
        "counts": counts,
        "rows_off_card": sum(1 for r in rows if r["on_card"] is False),
        "means": {key: round(statistics.fmean(r[key] for r in rows), 4)
                  for key in ("ours", "guest", "overlap", "code_share")},
        "correlations": {
            "ours_vs_guest": ours_guest,
            # the number was quoted with a band that no file held; now the file holds it
            "ours_vs_guest_ci95": rho_ci(rows, "ours", "guest"),
            "guest_vs_overlap": guest_overlap,
            "ours_vs_overlap": ours_overlap,
            "ours_vs_guest_given_overlap": controlled,
        },
        "strata": split,
        "predictions": {
            "guest_vs_overlap_below_0_3": None if guest_overlap is None
            else abs(guest_overlap) < RHO_WITH_OVERLAP,
            "partial_within_0_1_of_plain": None if controlled is None or ours_guest is None
            else abs(controlled - ours_guest) < PARTIAL_GAP,
            "strata_within_0_1": None if gap is None else gap < STRATUM_GAP,
            "n_at_least_100": len(rows) >= MIN_ROWS,
            "stratum_gap": None if gap is None else round(gap, 4),
        },
    }
