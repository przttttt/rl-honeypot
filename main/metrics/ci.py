from __future__ import annotations
import math
import numpy as np
from scipy import stats

def wilson_interval(successes: int | float, n: int, level: float=0.95) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    z = stats.norm.ppf(1 - (1 - level) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def bootstrap_ci(values: np.ndarray, level: float=0.95, resamples: int=10000, seed: int=0, statistic=np.mean) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(resamples, n))
    stats_samples = statistic(values[idx], axis=1)
    alpha = (1 - level) / 2
    (lo, hi) = np.quantile(stats_samples, [alpha, 1 - alpha])
    return (float(lo), float(hi))

def mean_with_ci(values: np.ndarray, level: float=0.95, resamples: int=10000, seed: int=0) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    (lo, hi) = bootstrap_ci(values, level, resamples, seed)
    return {'mean': float(values.mean()) if values.size else float('nan'), 'ci_lo': lo, 'ci_hi': hi, 'n': int(values.size)}

def auroc_ap(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind='mergesort')
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {'auroc': float('nan'), 'average_precision': float('nan')}
    ranks = _average_ranks(scores)
    sum_pos_ranks = float(ranks[y_true == 1].sum())
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    auroc = u / (n_pos * n_neg)
    tp = np.cumsum(y_sorted)
    precisions = tp / (np.arange(len(y_sorted)) + 1)
    ap = float((precisions * y_sorted).sum() / n_pos)
    return {'auroc': float(auroc), 'average_precision': ap}

def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0 or len(y_true) == 0:
        return float('nan')
    order = np.argsort(-np.asarray(scores), kind='mergesort')
    top = np.asarray(y_true)[order][:k]
    return float(top.sum() / min(k, len(y_true)))

def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - y_true) ** 2))
