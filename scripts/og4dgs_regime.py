"""Per-scene ``mu_t_lr`` and ``temporal_spread_init`` of the published 4DGS regime, for the og_4dgs baseline.

Published 4DGS works in world units and seconds; this repo normalizes the scene into a unit ball and time to a unit span.
The two quantities that do not survive that normalization are recomputed from the dataset:

* ``mu_t_lr``: the published temporal LR in seconds equals the spatial LR in world units,
  ``means_lr * cameras_extent``, divided by the sequence span.
* ``temporal_spread_init``: the published initial ``sigma_t = sqrt(span / 5)`` seconds, as a fraction of the span.

Usage::

    pixi run python scripts/og4dgs_regime.py data/4dgs/flame_steak_undistorted_r2
    pixi run python scripts/og4dgs_regime.py DIR --overrides   # Hydra overrides for `pixi run train`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


MEANS_LR = 1.6e-4  # training.means_lr of every shipped preset
NERFPP_RADIUS_SCALE = 1.1  # the published getNerfppNorm pads the camera radius by 10 %


def regime(data_dir: Path) -> dict[str, float]:
    """Published-regime numbers of one dataset, from its ``train_dataset.json``."""
    captures = json.loads((data_dir / "train_dataset.json").read_text())["captures"]
    centers = {cap["camera_id"]: cap["translation"] for cap in captures}  # poses are camera-to-world
    stacked = np.asarray(list(centers.values()))
    extent = NERFPP_RADIUS_SCALE * float(np.linalg.norm(stacked - stacked.mean(axis=0), axis=1).max())
    timestamps = sorted({cap["timestamp"] for cap in captures})
    span = timestamps[-1] - timestamps[0]
    if span <= 0:
        raise SystemExit(f"{data_dir} holds a single timestamp; the temporal regime needs a sequence")
    return {
        "mu_t_lr": MEANS_LR * extent / span,
        "temporal_spread_init": math.sqrt(span / 5.0) / span,
        "frames": len({cap["frame_id"] for cap in captures}),
    }


def main() -> None:
    """Print the regime of each dataset directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dirs", type=Path, nargs="+", help="dataset directories with a train_dataset.json")
    parser.add_argument("--overrides", action="store_true", help="print Hydra overrides instead of a table")
    args = parser.parse_args()
    if args.overrides and len(args.data_dirs) != 1:
        parser.error("--overrides takes exactly one dataset: the numbers belong to the data they train on")
    for data_dir in args.data_dirs:
        r = regime(data_dir)
        if args.overrides:
            print(f"training.mu_t_lr={r['mu_t_lr']:.6e} model.temporal_spread_init={r['temporal_spread_init']:.6f}")
        else:
            print(
                f"{data_dir.name:<36} frames={r['frames']:<5} mu_t_lr={r['mu_t_lr']:.4e} "
                f"temporal_spread_init={r['temporal_spread_init']:.4f}"
            )


if __name__ == "__main__":
    main()
