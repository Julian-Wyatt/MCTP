"""Token-impact maps at several pruning depths, replaying one shared region bank."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import torch

from models.token_pruning.prunable_classifier import PreparedState, PrunableClassifier
from utils.metrics.token_impact import estimate_token_impacts


@dataclass(frozen=True)
class TokenImpactMaps:
    impacts: torch.Tensor  # [B, D, N] float32; NaN where undefined (protected or never dropped/kept)
    protected: torch.Tensor  # [B, D, N] bool; tokens added to every draw at that depth
    targets: torch.Tensor  # [B] unpruned top-1 class y_hat that every draw is scored against
    predictions: torch.Tensor  # [B, D, M] top-1 class of each pruned draw


def top_norm_tokens(state: PreparedState, num_prefix_tokens: int, count: int) -> torch.Tensor:
    """[B, N] bool marking the `count` patch tokens with the largest l2 norm entering block state.depth."""
    norms = state.tokens[:, num_prefix_tokens:].float().norm(dim=-1)  # [B, N, C] -> [B, N]
    top = norms.argsort(dim=-1, descending=True, stable=True)[:, :count]
    return torch.zeros_like(norms, dtype=torch.bool).scatter_(1, top, True)


def random_tokens(filenames: list[str], num_tokens: int, count: int, seed: int) -> torch.Tensor:
    """[B, N] bool marking `count` uniformly random patch tokens per image.

    The choice depends only on (seed, filename), so it is the same at every
    depth and in every batch.
    """
    mask = torch.zeros(len(filenames), num_tokens, dtype=torch.bool)
    for row, filename in enumerate(filenames):
        image_seed = int.from_bytes(hashlib.sha256(f"{seed}:{filename}".encode()).digest()[:8], "little")
        mask[row, random.Random(image_seed).sample(range(num_tokens), count)] = True
    return mask


@torch.inference_mode()
def replay_keep_masks(
    model: PrunableClassifier,
    state: PreparedState,
    keep_masks: torch.Tensor,
    targets: torch.Tensor,
    max_batch: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run every (image, draw) pair from the cached state.

    keep_masks [B, M, N] bool; targets [B]. Pairs are batched by their kept-token
    count, so no padding is needed. Returns p_yhat [M, B] (float32) and the
    predicted class [M, B].
    """
    batch, draws, tokens = keep_masks.shape
    masks = keep_masks.cpu().reshape(batch * draws, tokens)
    counts = masks.sum(-1)
    probabilities = torch.empty(batch * draws, dtype=torch.float32)
    predictions = torch.empty(batch * draws, dtype=torch.long)
    device = state.tokens.device
    for count in counts.unique(sorted=True):
        for pairs in torch.where(counts == count)[0].split(max_batch):
            rows = (pairs // draws).to(device)  # image index of each (image, draw) pair
            pair_state = PreparedState(state.tokens.index_select(0, rows), state.positions, state.depth)
            logits = model.forward_kept(pair_state, masks[pairs].to(device)).float()
            probabilities[pairs] = logits.softmax(-1).gather(1, targets[rows, None]).squeeze(1).cpu()
            predictions[pairs] = logits.argmax(-1).cpu()
    # [B * M] -> [M, B]
    return probabilities.reshape(batch, draws).T, predictions.reshape(batch, draws).T


@torch.inference_mode()
def token_impact_maps(
    model: PrunableClassifier,
    images: torch.Tensor,
    *,
    depths: list[int],
    bank: torch.Tensor,
    intervened: PrunableClassifier | None = None,
    protected: torch.Tensor | None = None,
    protect_top_norm: int = 0,
    max_batch: int = 128,
) -> TokenImpactMaps:
    """Delta_i at each pruning depth for a batch of images.

    - **Model and target:** `model`'s unpruned top-1 is y_hat. The draws run on
      `intervened` if given (e.g. prefix removal, a test-time register),
      otherwise on `model`.
    - **Fixed protection:** `protected` [B, N] keeps those patch tokens in every
      draw at every depth.
    - **Top-norm protection:** `protect_top_norm` = k instead keeps the k
      highest-norm patch tokens entering each block l.
    - **Precision:** the unpruned prediction and the blocks before each depth
      run in float32. On CUDA, only the pruned draws run under bfloat16
      autocast.
    """
    if list(depths) != sorted(set(depths)):
        raise ValueError(f"depths must be strictly increasing, got {depths}.")
    pruned_model = model if intervened is None else intervened
    targets = model.forward_unpruned(images).argmax(-1)
    state = pruned_model.prepare(images, 0)
    if state.tokens.shape[1] - pruned_model.num_prefix_tokens != bank.shape[1]:
        raise ValueError(f"Region bank has {bank.shape[1]} tokens, but the images give "
                         f"{state.tokens.shape[1] - pruned_model.num_prefix_tokens} patch tokens.")
    impacts, protected_masks, predictions = [], [], []
    for depth in depths:
        state = pruned_model.advance(state, depth)
        if protect_top_norm:
            depth_protected = top_norm_tokens(state, pruned_model.num_prefix_tokens, protect_top_norm).cpu()
        elif protected is not None:
            depth_protected = protected.cpu()
        else:
            depth_protected = torch.zeros(images.shape[0], bank.shape[1], dtype=torch.bool)
        keep_masks = bank[None] | depth_protected[:, None]  # [M, N] -> [B, M, N]
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=images.is_cuda):
            probabilities, draw_predictions = replay_keep_masks(pruned_model, state, keep_masks, targets, max_batch)
        impacts.append(estimate_token_impacts(keep_masks.transpose(0, 1), probabilities))
        protected_masks.append(depth_protected)
        predictions.append(draw_predictions.T)
    return TokenImpactMaps(
        impacts=torch.stack(impacts, dim=1),
        protected=torch.stack(protected_masks, dim=1),
        targets=targets.cpu(),
        predictions=torch.stack(predictions, dim=1),
    )
