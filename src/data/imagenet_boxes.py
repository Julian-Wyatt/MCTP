"""ImageNet validation bounding boxes, projected onto the model's patch-token grid.

XML boxes use one-based inclusive pixel coordinates. Internally rectangles are
zero-based and half-open, so [xmin - 1, xmax) includes the last pixel.
"""

from __future__ import annotations

import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.transforms import GRID_SIZE, IMAGE_SIZE


@dataclass(frozen=True)
class ImageNetBoxes:
    width: int
    height: int
    objects: tuple[tuple[str, tuple[float, float, float, float]], ...]  # (WNID, (left, top, right, bottom))


def _parse_annotation(content: bytes) -> tuple[str, ImageNetBoxes]:
    annotation = ET.fromstring(content)
    filename = Path(annotation.findtext("filename", "")).stem
    width = int(annotation.findtext("size/width", "0"))
    height = int(annotation.findtext("size/height", "0"))
    if not filename or min(width, height) <= 0:
        raise ValueError(f"Bounding box XML without a filename or size: {filename!r}.")
    objects = []
    for obj in annotation.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin, ymin, xmax, ymax = (float(box.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
        edges = (max(0.0, xmin - 1), max(0.0, ymin - 1), min(float(width), xmax), min(float(height), ymax))
        if not np.isfinite([xmin, ymin, xmax, ymax]).all() or edges[2] <= edges[0] or edges[3] <= edges[1]:
            raise ValueError(f"Invalid bounding box for {filename}.")
        objects.append((str(obj.findtext("name", "")), edges))
    return filename + ".JPEG", ImageNetBoxes(width, height, tuple(objects))


def load_validation_boxes(path: Path) -> dict[str, ImageNetBoxes]:
    """Read the official ILSVRC2012_bbox_val_v3.tgz (or an extracted XML directory) without unpacking it."""
    annotations = {}

    def add(content: bytes) -> None:
        filename, boxes = _parse_annotation(content)
        if filename in annotations:
            raise ValueError(f"Duplicate bounding box annotation: {filename}")
        annotations[filename] = boxes

    if path.is_dir():
        for xml in sorted(path.rglob("*.xml")):
            add(xml.read_bytes())
    else:
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith(".xml"):
                    with archive.extractfile(member) as stream:
                        add(stream.read())
    if not annotations:
        raise ValueError(f"No bounding box XML annotations in {path}.")
    return annotations


def token_box_union(
    annotation: ImageNetBoxes, wnid: str, *, image_size: int = IMAGE_SIZE, grid_size: int = GRID_SIZE
) -> tuple[np.ndarray, int]:
    """Return a row-major [grid_size^2] mask of token centres inside any `wnid` box, and the visible box count.

    Reproduces `data.transforms.image_transform`: the shorter side is resized to
    image_size (the longer side truncated to int), then centre-cropped with
    Python round, as torchvision does. Boxes cropped out entirely are not visible.
    """
    if annotation.width <= annotation.height:
        resized_w, resized_h = image_size, int(image_size * annotation.height / annotation.width)
    else:
        resized_w, resized_h = int(image_size * annotation.width / annotation.height), image_size
    offset_x = int(round((resized_w - image_size) / 2))
    offset_y = int(round((resized_h - image_size) / 2))
    scale_x, scale_y = resized_w / annotation.width, resized_h / annotation.height
    # Token centres in the cropped input's pixel frame: [grid_size] per axis.
    centres = (np.arange(grid_size) + 0.5) * image_size / grid_size
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    visible_boxes = 0
    for name, (xmin, ymin, xmax, ymax) in annotation.objects:
        if name != wnid:
            continue
        left, right = max(0.0, xmin * scale_x - offset_x), min(float(image_size), xmax * scale_x - offset_x)
        top, bottom = max(0.0, ymin * scale_y - offset_y), min(float(image_size), ymax * scale_y - offset_y)
        if left >= right or top >= bottom:
            continue
        visible_boxes += 1
        inside_x = (centres >= left) & (centres < right)
        inside_y = (centres >= top) & (centres < bottom)
        mask |= inside_y[:, None] & inside_x[None, :]
    return mask.reshape(-1), visible_boxes
