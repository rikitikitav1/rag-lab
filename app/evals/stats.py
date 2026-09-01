import statistics

import numpy as np
from scipy.stats import wilcoxon

BOOTSTRAP_N = 10_000


def delta_stats(deltas: list, rng=None) -> dict:
    rng = rng if rng is not None else np.random.default_rng(42)
    arr = np.array(deltas, dtype=float)
    boot_means = rng.choice(arr, size=(BOOTSTRAP_N, arr.size), replace=True).mean(axis=1)
    p = 1.0 if np.all(arr == 0) else float(wilcoxon(arr).pvalue)
    return {
        "mean_delta": round(float(arr.mean()), 3),
        "ci95": [
            round(float(np.percentile(boot_means, 2.5)), 3),
            round(float(np.percentile(boot_means, 97.5)), 3),
        ],
        "p": round(p, 4),
        "n": int(arr.size),
    }


# reject while p(i) <= alpha/(m-i) and stop at the first failure, which holds the rest
def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    kept = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] > alpha / (len(pvalues) - rank):
            break
        kept[i] = True
    return kept


# None past the break: a test the step-down never reached had no bar to fail
def holm_thresholds(pvalues: list[float], alpha: float = 0.05) -> list[float | None]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out: list[float | None] = [None] * len(pvalues)
    for rank, i in enumerate(order):
        threshold = alpha / (len(pvalues) - rank)
        out[i] = threshold
        if pvalues[i] > threshold:
            break
    return out


# both reports wrote the same three keys onto every test and returned the same four
def annotate_holm(tests: list[dict], family: str, alpha: float = 0.05) -> dict:
    pvalues = [t["p"] for t in tests]
    kept, thresholds = holm(pvalues, alpha), holm_thresholds(pvalues, alpha)
    for test, keep, threshold in zip(tests, kept, thresholds, strict=True):
        # `<=`, the way `holm` reads its own threshold, or raw and corrected disagree at alpha
        test["significant_raw"] = test["p"] <= alpha
        test["significant_holm"] = keep
        test["holm_threshold"] = None if threshold is None else round(threshold, 5)
    return {"method": "holm", "alpha": alpha, "tests": len(tests), "family": family}


# a verdict column is a string in the database, so reading one is a cast
def score_of(value) -> int | None:
    return None if value is None else int(value)


def mean_of(values, digits: int = 2) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return round(statistics.fmean(kept), digits) if kept else None


# the ids come from the caller: their order is what a fixed seed drew indices into
def deltas_over(before: dict, after: dict, ids) -> list[float]:
    return [
        after[i] - before[i]
        for i in ids
        if before.get(i) is not None and after.get(i) is not None
    ]


# better, worse and the rest, counted once: three modules counted the two directions by hand
def tally(deltas) -> dict:
    better = sum(1 for d in deltas if d > 0)
    worse = sum(1 for d in deltas if d < 0)
    return {"better": better, "worse": worse, "unchanged": len(deltas) - better - worse}
