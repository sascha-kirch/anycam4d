"""Render a global- vs rolling-shutter side-by-side video from one checkpoint while the camera pans.

The camera moves on a small arc around an eval pose: it dollies ``--dolly`` · sin(2π t / ``--pan-period-s``) sideways
and yaws ``--pan-deg`` · sin(…) about its own vertical axis, over a ``--seconds`` window of the scene (both 0: static
camera, scene motion only). Rolling shutter: rows read out
top to bottom over ``--readout-ms``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from anycam4d.eval.scene_view import load_eval_view
from anycam4d.eval.scene_view import to_bgr8
from anycam4d.renderer.rolling_shutter import render_rolling_shutter
from anycam4d.renderer.rolling_shutter import row_viewmats
from anycam4d.renderer.ut_fused_backend import UTFusedBackend


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera


def moved(viewmat: Float[Tensor, "4 4"], deg: float, dolly: float) -> Float[Tensor, "4 4"]:
    """``viewmat`` dollied by ``dolly`` (scene units) along the camera's x axis, then turned by ``deg`` about its y."""
    out = viewmat.clone()
    out[0, 3] -= dolly
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    turn = torch.eye(4, device=viewmat.device)
    turn[0, 0], turn[0, 2], turn[2, 0], turn[2, 2] = c, -s, s, c
    return turn @ out


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("data_dir")
    parser.add_argument("out", type=Path, help="Output .mp4")
    parser.add_argument("--index", type=int, default=0, help="Eval-split sample giving the start pose.")
    parser.add_argument("--readout-ms", type=float, default=50.0)
    parser.add_argument("--pan-deg", type=float, default=12.0, help="Swing amplitude.")
    parser.add_argument("--pan-period-s", type=float, default=0.5)
    parser.add_argument("--dolly", type=float, default=0.0, help="Sideways amplitude in normalized scene units.")
    parser.add_argument("--start-s", type=float, default=0.0, help="Capture second the window starts at.")
    parser.add_argument("--seconds", type=float, default=1.0, help="Capture seconds rendered.")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    """Write the side-by-side video."""
    args = parse_args()

    view = load_eval_view(args.checkpoint, args.data_dir, args.index)
    model, cfg = view.model, view.cfg
    duration = view.dataset.t_max - view.dataset.t_min
    readout_s = args.readout_ms / 1000.0
    readout_tau = readout_s / duration
    width, height = view.camera.width, view.camera.height

    def camera_at(viewmat: Float[Tensor, "4 4"]) -> BatchedCamera:
        return dataclasses.replace(view.camera, viewmats=viewmat[None])

    start = view.camera.viewmats[0]
    global_backend = UTFusedBackend(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * width, height))
    peak_rate = 2.0 * math.pi * args.pan_deg / args.pan_period_s

    def pose_at(t: float) -> Float[Tensor, "4 4"]:
        phase = math.sin(2.0 * math.pi * (t - args.start_s) / args.pan_period_s)
        return moved(start, args.pan_deg * phase, args.dolly * phase)

    label = f"rolling {args.readout_ms:g} ms, pan <= {peak_rate:.0f} deg/s"
    with torch.no_grad():
        for k in range(args.frames):
            t = args.start_s + (k + 0.5) / args.frames * args.seconds  # capture seconds since t_min
            tau = t / duration - 0.5
            centre = pose_at(t)
            camera = camera_at(centre)
            global_colors = global_backend.render(model, camera, tau, model.sh_degree).colors
            rows = row_viewmats(
                pose_at(t - 0.5 * readout_s),
                pose_at(t + 0.5 * readout_s),
                height,
            )
            rolling_colors, _ = render_rolling_shutter(
                model, camera, rows[None], tau, readout_tau, model.sh_degree, cfg
            )
            writer.write(np.concatenate([to_bgr8(global_colors, "global"), to_bgr8(rolling_colors, label)], axis=1))
    writer.release()
    print(f"Wrote {args.frames} frames to {args.out} (readout {readout_tau:.4g} tau of a {duration:.2f} s sequence)")


if __name__ == "__main__":
    main()
