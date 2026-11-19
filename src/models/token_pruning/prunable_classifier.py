"""Linear-probed frozen ViT whose patch tokens can be pruned once, at the input of block l."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.backbones import FrozenViT
from models.probe import load_probe


@dataclass(frozen=True)
class PreparedState:
    """Dense residual stream entering block `depth`, computed once per image and reused by every draw."""

    tokens: torch.Tensor  # [B, P + N, C]
    positions: torch.Tensor | None  # shared RoPE table [N, 2D], or None
    depth: int


class PrunableClassifier(nn.Module):
    """Frozen backbone + linear probe. Prefix tokens ([CLS], registers) are always kept."""

    def __init__(self, backbone: FrozenViT, head: nn.Linear, probe_readout: str):
        super().__init__()
        self.backbone, self.head, self.probe_readout = backbone, head, probe_readout
        self.num_prefix_tokens = backbone.num_prefix_tokens

    @classmethod
    def from_probe(cls, backbone: FrozenViT, probe_path: str) -> PrunableClassifier:
        head, readout = load_probe(probe_path, backbone)
        return cls(backbone, head, readout)

    def global_token(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone.global_token(tokens)

    def readout(self, tokens: torch.Tensor) -> torch.Tensor:
        """Normalised [B, P + K, C] -> logits [B, classes]; GAP averages kept patch tokens only."""
        patch_mean = tokens[:, self.num_prefix_tokens :].mean(dim=1)
        if self.probe_readout == "gap":
            features = patch_mean.float()  # [B, C]
        else:
            features = torch.cat((self.global_token(tokens), patch_mean), dim=-1).float()  # [B, 2C]
        return self.head(features)

    def embed(self, images: torch.Tensor) -> PreparedState:
        tokens, positions = self.backbone.embed(images)
        return PreparedState(tokens, positions, 0)

    def prepare(self, images: torch.Tensor, depth: int) -> PreparedState:
        return self.advance(self.embed(images), depth)

    def advance(self, state: PreparedState, depth: int) -> PreparedState:
        """Run blocks state.depth .. depth - 1 on all tokens, so later depths reuse earlier ones."""
        if not state.depth <= depth <= len(self.backbone.blocks):
            raise ValueError(f"Cannot advance a state at depth {state.depth} to depth {depth}.")
        tokens = state.tokens
        for index in range(state.depth, depth):
            tokens = self.backbone.run_block(index, tokens, state.positions)
        return PreparedState(tokens, state.positions, depth)

    def forward_unpruned(self, images: torch.Tensor) -> torch.Tensor:
        state = self.prepare(images, len(self.backbone.blocks))
        return self.readout(self.backbone.normalize(state.tokens))

    def forward_kept(self, state: PreparedState, keep_mask: torch.Tensor) -> torch.Tensor:
        """Keep the prefix and the patch tokens where keep_mask [B, N] is True; run blocks depth..L-1.

        Every row must keep the same number K of patch tokens (the replay groups draws by K).
        """
        batch = state.tokens.shape[0]
        prefix, patches = state.tokens[:, : self.num_prefix_tokens], state.tokens[:, self.num_prefix_tokens :]
        # [B, N] bool -> [B, K] kept patch ids, ascending in each row.
        kept_ids = keep_mask.nonzero()[:, 1].reshape(batch, -1)
        # [B, N, C] -> [B, K, C]
        kept = patches.gather(1, kept_ids[..., None].expand(-1, -1, patches.shape[-1]))
        tokens = torch.cat((prefix, kept), dim=1)  # [B, P + K, C]
        positions = state.positions
        if positions is not None:
            positions = positions[kept_ids][:, None]  # [N, 2D] -> [B, 1, K, 2D], broadcast over heads
        for index in range(state.depth, len(self.backbone.blocks)):
            tokens = self.backbone.run_block(index, tokens, positions)
        return self.readout(self.backbone.normalize(tokens))
