"""Agreement between two token-impact maps of one image, e.g. at two pruning depths or from two seeds.

Both functions compare only tokens defined (finite) in both maps. To compare
several maps on one common token set (e.g. tokens unprotected at every depth),
set the excluded positions to NaN in every map first.
"""

from __future__ import annotations

import math

import numpy as np


def _defined_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.asarray(first, dtype=np.float64).reshape(-1), np.asarray(second, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ValueError(f"Maps must have the same number of tokens, got {first.size} and {second.size}.")
    defined = np.isfinite(first) & np.isfinite(second)
    return first[defined], second[defined]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks of a 1-D array; tied values share the mean of their ranks."""
    _, group, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)  # rank of the last member of each tie group, in sorted order
    return (last_rank - (counts - 1) / 2)[group]


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Tie-aware Spearman correlation of signed impacts; NaN if either map is constant."""
    first, second = _defined_pair(first, second)
    if first.size < 2:
        return float("nan")
    first, second = average_ranks(first), average_ranks(second)
    first, second = first - first.mean(), second - second.mean()
    denominator = math.sqrt(float(first @ first) * float(second @ second))
    return float("nan") if denominator == 0 else float((first @ second) / denominator)


def top_fraction_overlap(first: np.ndarray, second: np.ndarray, fraction: float = 0.1) -> float:
    """Share of the k = ceil(fraction * N) highest-impact tokens common to both maps.

    Random rankings score about `fraction`. Ties are broken towards the lower token index.
    """
    first, second = _defined_pair(first, second)
    if not first.size or not 0 < fraction <= 1:
        raise ValueError(f"Need at least one defined token and fraction in (0, 1], got {first.size} and {fraction}.")
    k = max(1, math.ceil(fraction * first.size))
    first_top = np.argsort(-first, kind="stable")[:k]
    second_top = np.argsort(-second, kind="stable")[:k]
    return float(np.intersect1d(first_top, second_top, assume_unique=True).size / k)
