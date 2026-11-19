"""Frozen pretrained ViTs, loaded from their original hubs and run one block at a time."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class BackboneSpec:
    loader: str  # "timm" or "siglip2"
    model_id: str
    num_prefix_tokens: int  # [CLS] + registers; never pruned
    readout: str  # probe input: "cls_gap" = [CLS ; GAP], "pool_gap" = [MAP pool ; GAP]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


BACKBONES = {
    "dinov3_vits16": BackboneSpec("timm", "vit_small_patch16_dinov3.lvd1689m", 5, "cls_gap", IMAGENET_MEAN, IMAGENET_STD),
    "dinov3_vitb16": BackboneSpec("timm", "vit_base_patch16_dinov3.lvd1689m", 5, "cls_gap", IMAGENET_MEAN, IMAGENET_STD),
    "dinov3_vitl16": BackboneSpec("timm", "vit_large_patch16_dinov3.lvd1689m", 5, "cls_gap", IMAGENET_MEAN, IMAGENET_STD),
    "siglip2_vitb16": BackboneSpec("siglip2", "google/siglip2-base-patch16-384", 0, "pool_gap", SIGLIP_MEAN, SIGLIP_STD),
}


class FrozenViT(nn.Module):
    """Token-level access to a frozen ViT: embed, one block, final norm, global token."""

    def __init__(self, name: str, img_size: int = 336):
        super().__init__()
        self.name, self.spec = name, BACKBONES[name]
        if self.spec.loader == "timm":
            import timm

            self.model = timm.create_model(
                self.spec.model_id, pretrained=True, num_classes=0, img_size=img_size, dynamic_img_size=True
            )
            self.blocks = self.model.blocks
            self.embed_dim = self.model.embed_dim
            loaded_prefix_tokens = self.model.num_prefix_tokens
        else:
            from transformers import SiglipVisionModel

            # The checkpoint stores a Conv2d patch projection, so it loads as SiglipVisionModel.
            self.model = SiglipVisionModel.from_pretrained(self.spec.model_id).vision_model
            self.blocks = self.model.encoder.layers
            self.embed_dim = self.model.config.hidden_size
            loaded_prefix_tokens = 0
        if loaded_prefix_tokens != self.spec.num_prefix_tokens:
            raise RuntimeError(f"{name}: expected {self.spec.num_prefix_tokens} prefix tokens, got {loaded_prefix_tokens}.")
        self.num_prefix_tokens = self.spec.num_prefix_tokens
        self.model.eval().requires_grad_(False)

    def embed(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """[B, 3, H, W] -> tokens [B, P + N, C], plus the RoPE table [N, 2D] (None without RoPE)."""
        if self.spec.loader == "siglip2":
            return self.model.embeddings(images, interpolate_pos_encoding=True), None
        tokens, rope = self.model._pos_embed(self.model.patch_embed(images))
        return self.model.norm_pre(tokens), rope

    def run_block(self, index: int, tokens: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        """positions: the shared [N, 2D] table, or [B, 1, K, 2D] rows for kept tokens; timm broadcasts both."""
        if self.spec.loader == "siglip2":
            return self.blocks[index](tokens, attention_mask=None)
        return self.blocks[index](tokens, rope=positions)

    def normalize(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.spec.loader == "siglip2":
            return self.model.post_layernorm(tokens)
        return self.model.norm(tokens)

    def global_token(self, tokens: torch.Tensor) -> torch.Tensor:
        """Normalised [B, P + K, C] -> [B, C]: [CLS], or SigLIP's attention-pooling (MAP) head over kept tokens."""
        if self.spec.loader == "siglip2":
            return self.model.head(tokens)
        return tokens[:, 0]
