"""Linear probes stored as safetensors, tagged with the backbone and readout they were trained on."""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import load_file, safe_open, save_file

from models.backbones import BACKBONES, FrozenViT


def save_probe(path: str, backbone: str, weight: torch.Tensor, bias: torch.Tensor, readout: str | None = None) -> None:
    """readout: the backbone's [global ; GAP] readout (default) or "gap" (mean of patch tokens only)."""
    save_file(
        {"weight": weight.contiguous(), "bias": bias.contiguous()},
        path,
        metadata={"backbone": backbone, "readout": readout or BACKBONES[backbone].readout},
    )


def load_probe(path: str, backbone: FrozenViT) -> tuple[nn.Linear, str]:
    """Return the probe and its readout: [global ; GAP] ([num_classes, 2C]) or "gap" ([num_classes, C])."""
    with safe_open(path, framework="pt") as file:
        metadata = file.metadata()
    readout = metadata.get("readout")
    if metadata.get("backbone") != backbone.name or readout not in (backbone.spec.readout, "gap"):
        raise ValueError(f"{path} holds a probe for {metadata}, not for {backbone.name}.")
    state = load_file(path)
    in_features = backbone.embed_dim * (1 if readout == "gap" else 2)
    if state["weight"].shape[1] != in_features:
        raise ValueError(f"{path}: probe input {state['weight'].shape[1]} != {in_features} for readout {readout}.")
    head = nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
    head.load_state_dict(state)
    return head.requires_grad_(False), readout
