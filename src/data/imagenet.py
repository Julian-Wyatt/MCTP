"""ImageNet-1k in the standard class-folder layout: <root>/<split>/<wnid>/<filename>.JPEG."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ImageNetSplit(Dataset):
    """Yields (image, label, filename). Labels index the sorted WNID folders, matching the probes' class order."""

    def __init__(self, root: str, split: str, transform):
        # allow_empty keeps class indices aligned on partial local copies with some empty class folders.
        self.folder = ImageFolder(Path(root) / split, transform=transform, allow_empty=True)
        self.wnids: list[str] = self.folder.classes
        if len(self.wnids) != 1000:
            raise ValueError(
                f"Expected 1000 class folders under {root}/{split}, found {len(self.wnids)}. Labels are indices into "
                "the sorted folder names, so every class folder must exist (empty folders are fine for partial copies)."
            )

    def __len__(self) -> int:
        return len(self.folder)

    def __getitem__(self, index: int):
        image, label = self.folder[index]
        return image, label, Path(self.folder.samples[index][0]).name
