"""Token impact Delta_i from Monte Carlo pruning draws."""

from __future__ import annotations

import torch


def estimate_token_impacts(
    keep_masks: torch.Tensor, target_probabilities: torch.Tensor
) -> torch.Tensor:
    """Mean p_yhat over draws that keep token i, minus the mean over draws that drop it.

    keep_masks: [M, B, N] bool, the bank mask for each draw (OR'd with any
        protected tokens), per image.
    target_probabilities: [M, B], p_yhat of the unpruned top-1 class in each draw.
    Returns [B, N] float32. Tokens kept in every draw or in none have no
    estimate and are NaN. This covers protected tokens and, at small M,
    tokens the bank never keeps. Downstream metrics must exclude NaN tokens.
    """

    if keep_masks.ndim != 3 or keep_masks.dtype != torch.bool:
        raise ValueError(
            f"keep_masks must be bool [M, B, N], got {keep_masks.dtype} {tuple(keep_masks.shape)}."
        )
    if target_probabilities.shape != keep_masks.shape[:2]:
        raise ValueError(
            f"target_probabilities {tuple(target_probabilities.shape)} must be [M, B] = {tuple(keep_masks.shape[:2])}."
        )
    keep = keep_masks.float()
    drop = 1.0 - keep
    objective = target_probabilities.float()[:, :, None]  # [M, B] -> [M, B, 1]
    # Sum over draws: [M, B, N] -> [B, N].
    keep_count, drop_count = keep.sum(0), drop.sum(0)
    kept_mean = (keep * objective).sum(0) / keep_count.clamp_min(1)
    dropped_mean = (drop * objective).sum(0) / drop_count.clamp_min(1)
    defined = (keep_count > 0) & (drop_count > 0)
    return (kept_mean - dropped_mean).masked_fill(~defined, torch.nan)
