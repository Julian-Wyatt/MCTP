"""Token-impact maps for one shard of ImageNet validation.

Run from the repository root (later YAML files and `section.key=value` override earlier ones):
    python src/core/run_mctp.py configs/mctp.yaml [more.yaml ...] [section.key=value ...]
Writes <run.output_dir>/shard<i>of<n>.npz.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from core.config import load_config, shard_path
from data.imagenet import ImageNetSplit
from data.transforms import NUM_PATCH_TOKENS, image_transform
from models.backbones import FrozenViT
from models.token_pruning.interventions import PrefixRemovedClassifier, TestTimeRegisterClassifier
from models.token_pruning.prunable_classifier import PrunableClassifier
from models.token_pruning.region_sampler import build_region_mask_bank
from trainer.mctp import random_tokens, token_impact_maps


def main() -> None:
    cfg = load_config(sys.argv[1:])
    device = torch.device(cfg.run.device)
    # The original GPU runs allowed bfloat16-internal float32 matmuls (no effect on CPU).
    torch.set_float32_matmul_precision("medium")

    dataset = ImageNetSplit(cfg.data.imagenet_root, "val", image_transform(cfg.model.backbone))
    # Shards interleave the (optionally truncated) split, so every shard sees every class.
    indices = list(range(cfg.data.max_images or len(dataset)))[cfg.run.shard :: cfg.run.num_shards]
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    model = PrunableClassifier.from_probe(FrozenViT(cfg.model.backbone), cfg.model.probe).to(device).eval()
    if model.head.out_features != len(dataset.wnids):
        raise ValueError(f"Probe predicts {model.head.out_features} classes, the dataset has {len(dataset.wnids)}.")
    intervened = None
    if cfg.mctp.intervention == "prefix_removed":
        intervened = PrefixRemovedClassifier(model).eval()
    elif cfg.mctp.intervention == "registers":
        intervened = TestTimeRegisterClassifier(model, cfg.mctp.register_neurons).eval()
    bank = build_region_mask_bank(
        num_samples=cfg.mctp.num_draws,
        num_tokens=NUM_PATCH_TOKENS,
        num_keep=round(cfg.mctp.keep_fraction * NUM_PATCH_TOKENS),
        region_sizes=cfg.mctp.region_sizes,
        seed=cfg.mctp.seed,
    )

    outputs = {key: [] for key in ("filenames", "labels", "targets", "impacts", "protected", "stability")}
    for batch, (images, labels, filenames) in enumerate(loader):
        protected = None
        if cfg.mctp.protection == "random":
            protected = random_tokens(list(filenames), NUM_PATCH_TOKENS, cfg.mctp.protected_tokens, cfg.mctp.protection_seed)
        maps = token_impact_maps(
            model,
            images.to(device),
            depths=cfg.mctp.depths,
            bank=bank,
            intervened=intervened,
            protected=protected,
            protect_top_norm=cfg.mctp.protected_tokens if cfg.mctp.protection == "top_norm" else 0,
            max_batch=cfg.mctp.replay_batch,
        )
        # Prediction stability: share of draws whose top-1 is still y_hat, [B, D, M] -> [B, D].
        stability = (maps.predictions == maps.targets[:, None, None]).float().mean(-1)
        outputs["filenames"].append(np.asarray(filenames))
        outputs["labels"].append(labels.numpy())
        outputs["targets"].append(maps.targets.numpy())
        outputs["impacts"].append(maps.impacts.numpy())
        outputs["protected"].append(maps.protected.numpy())
        outputs["stability"].append(stability.numpy())
        print(f"[MCTP] batch {batch + 1}/{len(loader)}", flush=True)

    path = shard_path(cfg, cfg.run.shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{key: np.concatenate(value) for key, value in outputs.items()},
        depths=np.asarray(cfg.mctp.depths),
        wnids=np.asarray(dataset.wnids),
        config=OmegaConf.to_yaml(cfg),
    )
    print(f"[MCTP] wrote {len(indices)} images to {path}")


if __name__ == "__main__":
    main()
