import statistics

import numpy as np
from scipy.stats import wilcoxon

BOOTSTRAP_N = 10_000


# both callers wrote this by hand, and this branch had to fix the rounding in both separately
def wilcoxon_p(deltas) -> float:
    return 1.0 if all(d == 0 for d in deltas) else float(wilcoxon(deltas).pvalue)


# the one resampling of the stand: two of them drew a different number of times and read as one
def bootstrap_ci(deltas, seed: int = 42, rng=None) -> tuple[float, float]:
    # sorted: the draw is over the values, and the caller's order must not move the interval
    arr = np.sort(np.array(list(deltas), dtype=float))
    rng = rng if rng is not None else np.random.default_rng(seed)
    means = rng.choice(arr, size=(BOOTSTRAP_N, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def delta_stats(deltas: list, rng=None) -> dict:
    arr = np.sort(np.array(deltas, dtype=float))
    low, high = bootstrap_ci(arr, rng=rng)
    p = wilcoxon_p(arr)
    return {
        "mean_delta": round(float(arr.mean()), 3),
        "ci95": [round(low, 3), round(high, 3)],
        # raw: `annotate_holm` decides on this, and rounding only ever lets a test past the bar
        "p": p,
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


# the judge's scale is 0..10 and every reader divided by ten by hand, in four places
def to_unit(value) -> float | None:
    got = score_of(value)
    return None if got is None else got / 10


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
