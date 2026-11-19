"""Choose the MLP neurons that a test-time register takes over, from ImageNet training images.

Run from the repository root with a SigLIP 2 config:
    python src/core/calibrate_registers.py configs/mctp.yaml model.backbone=siglip2_vitb16 model.probe=...
Writes <run.output_dir>/test_time_registers.yaml. Pass it as an extra config to
run_mctp.py to prune with the register.
"""

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from core.config import load_config
from data.imagenet import ImageNetSplit
from data.transforms import image_transform
from models.backbones import FrozenViT
from models.token_pruning.interventions import TestTimeRegisterClassifier
from models.token_pruning.prunable_classifier import PrunableClassifier
from trainer.register_calibration import rank_register_neurons, register_trial


def main() -> None:
    cfg = load_config(sys.argv[1:])
    settings = cfg.calibration
    device = torch.device(cfg.run.device)
    torch.set_float32_matmul_precision("medium")
    dataset = ImageNetSplit(cfg.data.imagenet_root, "train", image_transform(cfg.model.backbone))
    indices = sorted(random.Random(settings.seed).sample(range(len(dataset)), settings.images))
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    model = PrunableClassifier.from_probe(FrozenViT(cfg.model.backbone), cfg.model.probe).to(device).eval()

    def image_batches():
        return (images.to(device) for images, _, _ in loader)

    ranked = rank_register_neurons(
        model, image_batches(), outlier_threshold=settings.outlier_threshold, last_block=settings.last_block
    )
    trials = []
    for count in settings.top_k:
        registered = TestTimeRegisterClassifier(model, [(block, unit) for block, unit, _ in ranked[:count]]).eval()
        trial = register_trial(model, registered, image_batches(), outlier_threshold=settings.outlier_threshold)
        trials.append({"top_k": count, **trial})
        print(f"[registers] top_k={count}: {trial}", flush=True)
        del registered
    # Fewest images with outliers left, then most predictions preserved, then fewest neurons.
    chosen = min(trials, key=lambda trial: (trial["outlier_image_fraction"], -trial["prediction_agreement"], trial["top_k"]))
    neurons = [[block, unit] for block, unit, _ in ranked[: chosen["top_k"]]]
    path = Path(cfg.run.output_dir) / "test_time_registers.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml({"mctp": {"intervention": "registers", "register_neurons": neurons}}))
    print(f"[registers] chose top_k={chosen['top_k']}; wrote {path}")


if __name__ == "__main__":
    main()
