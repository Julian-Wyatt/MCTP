"""Calibrate test-time registers: which MLP neurons to move into the added register token.

1. Rank neurons by their mean |activation| at high-norm outlier tokens of the
   unpruned model (Jiang et al., Algorithm 1).
2. Try moving the top-k neurons for several k, and keep the k that removes the
   most outliers. Ties go to the k that best preserves the original
   predictions, then to the smallest k.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from models.token_pruning.interventions import TestTimeRegisterClassifier
from models.token_pruning.prunable_classifier import PrunableClassifier


@torch.inference_mode()
def rank_register_neurons(
    model: PrunableClassifier, image_batches: Iterable[torch.Tensor], *, outlier_threshold: float, last_block: int
) -> list[tuple[int, int, float]]:
    """Return (block, unit, score) for every MLP neuron in blocks 0..last_block, highest score first.

    Outliers are tokens of the final block output whose norm exceeds
    outlier_threshold x the image's median token norm. A neuron's score is its
    |activation| averaged over an image's outlier tokens, then over the images
    that have outliers.
    """
    if model.num_prefix_tokens:
        raise ValueError("Register neurons are ranked on backbones without prefix tokens.")
    totals: dict[int, torch.Tensor] = {}
    images_with_outliers = 0
    for images in image_batches:
        norms = model.prepare(images, len(model.backbone.blocks)).tokens.float().norm(dim=-1)  # [B, N]
        outliers = norms > outlier_threshold * norms.quantile(0.5, dim=-1, keepdim=True)
        present = outliers.any(-1)
        if not bool(present.any()):
            continue
        counts = outliers.sum(-1).clamp_min(1)

        def accumulate(block):
            def hook(module, args):
                # [B, N, H] -> [B, H] mean over outlier tokens -> [H] summed over images with outliers.
                per_image = (args[0].float().abs() * outliers[..., None]).sum(1) / counts[:, None]
                contribution = per_image[present].sum(0).double().cpu()
                totals[block] = totals.get(block, torch.zeros_like(contribution)) + contribution

            return hook

        handles = [model.backbone.blocks[block].mlp.fc2.register_forward_pre_hook(accumulate(block))
                   for block in range(last_block + 1)]
        try:
            model.prepare(images, last_block + 1)
        finally:
            for handle in handles:
                handle.remove()
        images_with_outliers += int(present.sum())
    if not images_with_outliers:
        raise ValueError("No calibration image has an outlier token.")
    ranked = [(block, unit, float(score / images_with_outliers))
              for block, scores in totals.items() for unit, score in enumerate(scores)]
    return sorted(ranked, key=lambda row: (-row[2], row[0], row[1]))


@torch.inference_mode()
def register_trial(
    model: PrunableClassifier,
    registered: TestTimeRegisterClassifier,
    image_batches: Iterable[torch.Tensor],
    *,
    outlier_threshold: float,
) -> dict[str, float]:
    """Share of images that still have an outlier patch token at any block output, and prediction agreement."""
    with_outliers, agreeing, total = 0, 0, 0
    for images in image_batches:
        state = registered.embed(images)
        has_outlier = torch.zeros(images.shape[0], dtype=torch.bool)
        for depth in range(1, len(registered.backbone.blocks) + 1):
            state = registered.advance(state, depth)
            # Patch tokens only (token 0 is the register): [B, 1 + N, C] -> [B, N].
            norms = state.tokens[:, 1:].float().norm(dim=-1).double().cpu()
            median = norms.median(dim=-1, keepdim=True).values
            ratios = torch.where(median > 0, norms / median, torch.zeros_like(norms))  # norm / median patch norm
            has_outlier |= (ratios > outlier_threshold).any(-1)
        predictions = registered.readout(registered.backbone.normalize(state.tokens)).argmax(-1)
        with_outliers += int(has_outlier.sum())
        agreeing += int((predictions == model.forward_unpruned(images).argmax(-1)).sum())
        total += images.shape[0]
    return {"outlier_image_fraction": with_outliers / total, "prediction_agreement": agreeing / total}
