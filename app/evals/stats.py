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
