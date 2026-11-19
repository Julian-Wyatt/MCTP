"""Effective support: how widely an image's positive token impact is spread across its tokens."""

from __future__ import annotations

import numpy as np


def positive_effective_support(impacts: np.ndarray) -> np.ndarray:
    """phi+ = exp(H+) / N for maps [..., N] -> [...].

    H+ is the entropy of q+_i = max(Delta_i, 0) / sum_j max(Delta_j, 0). phi+
    is 1 for positive impact spread evenly over all N tokens and 1/N when it
    sits on one token. Undefined (NaN) tokens are excluded, so N counts defined
    tokens. Maps with no positive impact, or fewer than two defined tokens,
    take H+ = 0 (phi+ = 1/N).
    """
    defined = np.isfinite(impacts)
    weights = np.where(defined, np.maximum(np.nan_to_num(impacts), 0), 0).astype(np.float64)
    total = weights.sum(axis=-1, keepdims=True)
    q = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    entropy = -np.sum(q * np.log(q, out=np.zeros_like(q), where=q > 0), axis=-1)
    count = defined.sum(axis=-1)
    entropy = np.where((total[..., 0] > 0) & (count >= 2), entropy, 0.0)
    return np.exp(entropy) / count
