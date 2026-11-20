"""Run configuration: a typed schema with the method's defaults, overridden by YAML files and CLI `section.key=value`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, OmegaConf

from models.backbones import BACKBONES


@dataclass
class DataConfig:
    imagenet_root: str = MISSING  # contains val/<wnid>/<filename>.JPEG
    boxes: str = ""  # ILSVRC2012_bbox_val_v3.tgz or its extracted XML directory; needed for off-box mass
    max_images: int = 0  # 0 = the whole validation split; N > 0 = its first N images (smoke runs)
    batch_size: int = 16
    num_workers: int = 8


@dataclass
class ModelConfig:
    backbone: str = "dinov3_vitb16"
    probe: str = MISSING  # linear probe .safetensors for this backbone


@dataclass
class MCTPConfig:
    depths: list[int] = field(default_factory=lambda: [0, 3, 6, 9])  # prune at the input of these blocks
    num_draws: int = 1000  # M
    keep_fraction: float = 0.1  # K = round(keep_fraction * N)
    region_sizes: list[int] = field(default_factory=lambda: [3, 7])
    seed: int = 42  # region bank seed; one bank shared by all images and depths
    replay_batch: int = 128  # (image, draw) pairs per forward
    # Patch tokens kept in every draw: none | top_norm (highest norm entering block l) | random (fixed per image).
    protection: str = "none"
    protected_tokens: int = 5
    protection_seed: int = 42  # random protection: the choice depends only on this seed and the filename
    # none | prefix_removed (prefix tokens dropped at block l; GAP probe) | registers (add a test-time register).
    intervention: str = "none"
    register_neurons: list[list[int]] = field(default_factory=list)  # [[block, unit], ...]; [] = empty register


@dataclass
class CalibrationConfig:
    """Test-time register calibration on ImageNet training images (src/core/calibrate_registers.py)."""

    images: int = 500
    seed: int = 42
    top_k: list[int] = field(default_factory=lambda: [10, 20, 50, 100])  # neuron counts to try
    outlier_threshold: float = 5.0  # outlier token: norm > threshold x median token norm
    last_block: int = 9  # rank MLP neurons in blocks 0..last_block


@dataclass
class RunConfig:
    output_dir: str = "outputs/mctp"
    device: str = "cuda"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mctp: MCTPConfig = field(default_factory=MCTPConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_config(arguments: list[str]) -> Config:
    """Merge schema defaults <- each YAML file in order <- `section.key=value` overrides.

    Unknown keys and missing values raise.
    """
    files = [argument for argument in arguments if argument.endswith((".yaml", ".yml"))]
    overrides = [argument for argument in arguments if not argument.endswith((".yaml", ".yml"))]
    merged = OmegaConf.merge(
        OmegaConf.structured(Config), *map(OmegaConf.load, files), OmegaConf.from_dotlist(overrides)
    )
    cfg: Config = OmegaConf.to_object(merged)
    if cfg.model.backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {cfg.model.backbone!r}; choose from {sorted(BACKBONES)}.")
    if not cfg.mctp.depths or cfg.mctp.depths != sorted(set(cfg.mctp.depths)) or cfg.mctp.depths[0] < 0:
        raise ValueError(f"mctp.depths must be non-negative and strictly increasing, got {cfg.mctp.depths}.")
    if not 0 < cfg.mctp.keep_fraction < 1 or cfg.mctp.num_draws <= 0 or cfg.mctp.protected_tokens <= 0:
        raise ValueError("mctp.keep_fraction must lie in (0, 1), num_draws > 0 and protected_tokens > 0.")
    if cfg.mctp.protection not in ("none", "top_norm", "random"):
        raise ValueError(f"mctp.protection must be none, top_norm or random, got {cfg.mctp.protection!r}.")
    has_prefix = BACKBONES[cfg.model.backbone].num_prefix_tokens > 0
    if cfg.mctp.intervention not in ("none", "prefix_removed", "registers"):
        raise ValueError(f"mctp.intervention must be none, prefix_removed or registers, got {cfg.mctp.intervention!r}.")
    if (cfg.mctp.intervention == "prefix_removed" and not has_prefix) or (cfg.mctp.intervention == "registers" and has_prefix):
        raise ValueError("prefix_removed needs a backbone with prefix tokens (DINOv3); registers one without (SigLIP 2).")
    return cfg


def maps_path(cfg: Config) -> Path:
    return Path(cfg.run.output_dir) / "token_impact_maps.npz"
