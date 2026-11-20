"""Summarise a finished MCTP run: prediction stability and off-box mass at each pruning depth.

Run from the repository root with the same configs and overrides as run_mctp.py:
    python src/core/score_mctp.py configs/mctp.yaml [more.yaml ...] [section.key=value ...]
Reads <run.output_dir>/token_impact_maps.npz and writes summary.json beside it.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from omegaconf import OmegaConf

from core.config import load_config, maps_path
from data.imagenet_boxes import load_validation_boxes
from utils.metrics.off_box_mass import off_box_masses


def main() -> None:
    cfg = load_config(sys.argv[1:])
    with np.load(maps_path(cfg)) as saved:
        run = {key: saved[key] for key in ("filenames", "labels", "targets", "impacts", "stability", "depths", "wnids", "config")}
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
    output.write_text(json.dumps({"config": OmegaConf.to_container(OmegaConf.create(str(run["config"]))), "depths": rows}, indent=2))
    print(f"[MCTP] wrote {output}")


if __name__ == "__main__":
    main()
