"""Evaluation preprocessing shared by every backbone."""

from __future__ import annotations

from torchvision import transforms

from models.backbones import BACKBONES

IMAGE_SIZE = 336
GRID_SIZE = IMAGE_SIZE // 16  # every supported backbone uses 16-pixel patches: a 21 x 21 token grid
NUM_PATCH_TOKENS = GRID_SIZE * GRID_SIZE


def image_transform(backbone: str, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Bicubic resize of the shorter side to image_size, centre crop, normalise with the backbone's statistics.

    `data.imagenet_boxes.token_box_union` reproduces this geometry; change both together.
    """
    spec = BACKBONES[backbone]
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=spec.mean, std=spec.std),
        ]
    )
