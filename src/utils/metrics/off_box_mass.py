"""Off-box mass: how much positive token impact lies outside the ground-truth object boxes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from data.imagenet_boxes import ImageNetBoxes, token_box_union


def normalised_off_box_mass(impacts: np.ndarray, inside: np.ndarray) -> float | None:
    """N * sum_{i outside} q+_i / (N - |inside|), with q+_i = max(Delta_i, 0) / sum_j max(Delta_j, 0).

    Uniform positive impact scores 1, impact only inside the boxes scores 0.
    Undefined (NaN) tokens, e.g. protected ones, are excluded from both the
    mass and the area. Returns None if there is no positive impact or no
    defined token outside the boxes.
    """
    defined = np.isfinite(impacts)
    impacts, inside = np.asarray(impacts, dtype=np.float64)[defined], inside[defined]
    positive = np.maximum(impacts, 0)
    total, outside_tokens = float(positive.sum()), int((~inside).sum())
    if total <= 0 or outside_tokens == 0:
        return None
    raw_mass = float(positive[~inside].sum() / total)
    return impacts.size * raw_mass / outside_tokens


def off_box_masses(
    filenames: Sequence[str],
    labels: np.ndarray,
    targets: np.ndarray,
    impacts: np.ndarray,
    wnids: Sequence[str],
    annotations: Mapping[str, ImageNetBoxes],
) -> np.ndarray:
    """Per-image off-box mass [I], NaN where undefined; report the mean over the finite entries.

    labels [I] are ground-truth class indices into `wnids`, targets [I] the unpruned
    top-1 predictions and impacts [I, N] the token-impact maps.
    Only correctly classified images are scored, so the impact explains the
    annotated class. Boxes of that class are merged into one token mask.
    """
    grid_size = int(np.sqrt(impacts.shape[1]))
    masses = np.full(len(filenames), np.nan)
    for index, (filename, label, target) in enumerate(zip(filenames, labels, targets, strict=True)):
        if int(target) != int(label) or filename not in annotations:
            continue
        inside, visible_boxes = token_box_union(annotations[filename], wnids[label], grid_size=grid_size)
        if visible_boxes == 0:
            continue
        mass = normalised_off_box_mass(impacts[index], inside)
        if mass is not None:
            masses[index] = mass
    return masses
