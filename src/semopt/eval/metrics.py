"""Evaluation metrics (SRS §9, Metric 2 + Metric 3).

Task-specific scoring:

* ``binary_f1`` — precision/recall/F1 for ``sem_filter`` (boolean predictions).
* ``exact_match`` — fraction of exact string matches for ``sem_map``.
* ``span_f1`` — token-overlap F1 for ``sem_extract`` fields (order-insensitive).
* ``ndcg_at_k`` — ranking quality for ``sem_rank`` against graded relevance.

Plus ``bootstrap_ci`` — the 95% bootstrap confidence interval used for every headline
number (Metric 3 / statistical protocol: never a single-run point estimate).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float


def binary_f1(y_true: Sequence[bool], y_pred: Sequence[bool]) -> PRF:
    """Precision/recall/F1 for the positive (True) class."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if p and not t)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1)


def accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true)


def exact_match(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Fraction of predictions equal to gold after whitespace/case normalization."""
    def norm(s: str) -> str:
        return " ".join(s.strip().lower().split())

    return accuracy([norm(g) for g in golds], [norm(p) for p in preds])


def span_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between two strings (order-insensitive, multiset-aware)."""
    pt = Counter(pred.lower().split())
    gt = Counter(gold.lower().split())
    if not pt and not gt:
        return 1.0
    overlap = sum((pt & gt).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pt.values())
    recall = overlap / sum(gt.values())
    return 2 * precision * recall / (precision + recall)


def mean_span_f1(preds: Sequence[str], golds: Sequence[str]) -> float:
    if len(preds) != len(golds):
        raise ValueError("length mismatch")
    if not preds:
        return 0.0
    return float(np.mean([span_f1(p, g) for p, g in zip(preds, golds, strict=True)]))


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    return sum(
        rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(ranked_relevances: Sequence[float], k: int) -> float:
    """NDCG@k: DCG of the produced ranking over DCG of the ideal ranking."""
    dcg = dcg_at_k(ranked_relevances, k)
    ideal = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI (default 95%) for the mean of ``values`` (Metric 3)."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = [arr[rng.integers(0, n, n)].mean() for _ in range(n_resamples)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
