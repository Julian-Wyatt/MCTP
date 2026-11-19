"""Summarise a finished MCTP run: prediction stability and off-box mass at each pruning depth.

Run from the repository root with the same configs and overrides as run_mctp.py:
    python src/core/score_mctp.py configs/mctp.yaml [more.yaml ...] [section.key=value ...]
Reads every <run.output_dir>/shard<i>of<n>.npz and writes summary.json beside them.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from omegaconf import OmegaConf

from core.config import load_config, shard_path
from data.imagenet_boxes import load_validation_boxes
from utils.metrics.off_box_mass import off_box_masses


def load_shards(cfg) -> dict[str, np.ndarray]:
    """Concatenate all shards of the run, rejecting missing shards or shards produced with other settings."""
    expected = OmegaConf.to_container(OmegaConf.structured(cfg))
    arrays: dict[str, list] = {}
    for shard in range(cfg.run.num_shards):
        path = shard_path(cfg, shard)
        if not path.exists():
            raise FileNotFoundError(f"Missing shard {path}; run run_mctp.py with run.shard={shard}.")
        with np.load(path) as saved:
            produced = OmegaConf.to_container(OmegaConf.create(str(saved["config"])))
            produced["run"]["shard"] = expected["run"]["shard"]
            if produced != expected:
                raise ValueError(f"{path} was produced with a different config.")
            for key in ("filenames", "labels", "targets", "impacts", "stability"):
                arrays.setdefault(key, []).append(saved[key])
            depths, wnids = saved["depths"], saved["wnids"]
    return {key: np.concatenate(value) for key, value in arrays.items()} | {"depths": depths, "wnids": wnids}


def main() -> None:
    cfg = load_config(sys.argv[1:])
    run = load_shards(cfg)
    annotations = load_validation_boxes(Path(cfg.data.boxes)) if cfg.data.boxes else None
    rows = []
    for index, depth in enumerate(run["depths"].tolist()):
        row = {"depth": depth, "images": len(run["filenames"]), "stability": float(run["stability"][:, index].mean())}
        if annotations is not None:
            masses = off_box_masses(run["filenames"].tolist(), run["labels"], run["targets"],
                                    run["impacts"][:, index], run["wnids"].tolist(), annotations)
            row |= {"off_box_mass": float(np.nanmean(masses)), "off_box_images": int(np.isfinite(masses).sum())}
        rows.append(row)
        print("  ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))
    output = Path(cfg.run.output_dir) / "summary.json"
    output.write_text(json.dumps({"config": OmegaConf.to_container(OmegaConf.structured(cfg)), "depths": rows}, indent=2))
    print(f"[MCTP] wrote {output}")


if __name__ == "__main__":
    main()
