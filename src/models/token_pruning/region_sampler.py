"""Seeded region sampler Q_K for Monte Carlo Token Pruning.

One bank of M keep masks is drawn once per run and shared by every image and
every pruning depth, so depth-wise comparisons use identical draws.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def _clipped_region(centre: int, side: int, size: int) -> tuple[int, ...]:
    """Token ids of the size x size square around `centre` on a side x side grid, clipped at the border."""
    row, column = divmod(centre, side)
    before, after = (size - 1) // 2, size - (size - 1) // 2
    rows = range(max(0, row - before), min(side, row + after))
    columns = range(max(0, column - before), min(side, column + after))
    return tuple(r * side + c for r in rows for c in columns)


def build_region_mask_bank(
    *,
    num_samples: int,
    num_tokens: int,
    num_keep: int,
    region_sizes: Sequence[int],
    seed: int,
) -> torch.Tensor:
    """Return [M, N] bool keep masks, each a union of clipped, non-overlapping n x n regions.

    For each draw: choose n uniformly from `region_sizes`, set the budget
    B = max(K, n^2), then keep adding a uniformly chosen feasible region until
    fewer than n^2 tokens of budget remain or nothing fits. A region is feasible
    if it does not overlap the retained tokens and fits in the remaining budget.

    The sequence of generator calls defines the bank: keep the order of choices
    and the centre order (row-major) unchanged, or every token-impact map changes.
    """

    side = math.isqrt(num_tokens)
    sizes = tuple(int(size) for size in region_sizes)
    if side * side != num_tokens:
        raise ValueError(f"Region sampling needs a square token grid, got {num_tokens} tokens.")
    if not 0 < num_keep < num_tokens or not sizes or not all(0 < size <= side for size in sizes):
        raise ValueError(f"Invalid budget K={num_keep} or region sizes {sizes} for a {side}x{side} grid.")

    # A CPU generator gives the same bank on every machine; CUDA generators produce a different stream.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def choose_uniformly(options):
        return options[int(torch.randint(len(options), (), generator=generator))]

    regions = {size: [_clipped_region(centre, side, size) for centre in range(num_tokens)] for size in sizes}
    masks = torch.zeros(num_samples, num_tokens, dtype=torch.bool)
    for draw in range(num_samples):
        size = choose_uniformly(sizes)
        budget = max(num_keep, size * size)
        retained: set[int] = set()
        while budget - len(retained) >= size * size:
            feasible = [
                region
                for region in regions[size]
                if len(region) <= budget - len(retained) and retained.isdisjoint(region)
            ]
            if not feasible:
                break
            retained.update(choose_uniformly(feasible))
        masks[draw, list(retained)] = True
    return masks
