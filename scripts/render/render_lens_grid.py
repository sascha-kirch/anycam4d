"""Render one checkpoint through four lenses from one pose while the scene plays: a 2 x 2 grid video.

Usage: ``python scripts/render/render_lens_grid.py <checkpoint.pt> <data_dir> <out.mp4> [--index 0]``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.registry import distortion_width
from anycam4d.camera.validity import compute_validity
from anycam4d.enums import CameraModelType
from anycam4d.eval.scene_view import load_eval_view
from anycam4d.eval.scene_view import to_bgr8
from anycam4d.renderer.ut_fused_backend import UTFusedBackend


if TYPE_CHECKING:
    from jaxtyping import Float
    from jaxtyping import UInt8
    from torch import Tensor


TILE_H = 540
TILE_W = 960


def lens(
    model: CameraModelType,
    focal: float,
    width: int,
    viewmat: Float[Tensor, "4 4"],
    coefficients: tuple[float, ...] = (),
) -> BatchedCamera:
    """Centred camera of ``model`` at ``viewmat``, ``TILE_H`` tall; ``coefficients`` fill the leading distortion slots."""
    device = viewmat.device
    K = torch.tensor([[focal, 0.0, width / 2], [0.0, focal, TILE_H / 2], [0.0, 0.0, 1.0]], device=device)
    distortion = torch.zeros(1, distortion_width(model), device=device)
    distortion[0, : len(coefficients)] = torch.tensor(coefficients, device=device)
    return BatchedCamera(
        viewmats=viewmat[None], Ks=K[None], model=model, width=width, height=TILE_H, distortion=distortion
    )


def tile(colors: Float[Tensor, "1 H W 3"], label: str) -> UInt8[np.ndarray, "TH TW 3"]:
    """Letterbox one render into a ``TILE_W`` x ``TILE_H`` tile with its label."""
    image = to_bgr8(colors)
    height = TILE_H
    width = round(image.shape[1] * height / image.shape[0])
    if image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    out = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    left = (TILE_W - width) // 2
    out[:, left : left + width] = image
    for color, thickness in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(out, label, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness, cv2.LINE_AA)
    return out


def main() -> None:
    """Write the grid video."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("data_dir")
    parser.add_argument("out", type=Path)
    parser.add_argument("--index", type=int, default=0, help="Eval-split sample giving the pose.")
    parser.add_argument("--start-s", type=float, default=0.0, help="Capture second the window starts at.")
    parser.add_argument("--seconds", type=float, default=4.0, help="Capture seconds rendered.")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    view = load_eval_view(args.checkpoint, args.data_dir, args.index)
    viewmat = view.camera.viewmats[0]
    duration = view.dataset.t_max - view.dataset.t_min
    if duration > 0.0 and args.start_s + args.seconds > duration:
        raise SystemExit(f"--start-s + --seconds exceeds the {duration:.2f} s capture; the model is untrained there")

    equidistant = lens(CameraModelType.FISHEYE_EQUIDISTANT, TILE_H / math.pi, TILE_H, viewmat)
    circle = compute_validity(
        equidistant.model, equidistant.Ks[0], equidistant.distortion[0], TILE_H, TILE_H, max_incidence_deg=90.0
    )
    circle = circle.to(viewmat.device)[None, ..., None]
    views = [
        ("Pinhole 90 deg", lens(CameraModelType.PINHOLE, TILE_H / 2, TILE_H, viewmat), None),
        (
            "Brown-Conrady, barrel",
            lens(CameraModelType.OPENCV, TILE_H * 2 / 3, TILE_H * 4 // 3, viewmat, (-0.28, 0.06)),
            None,
        ),
        (f"Training lens ({view.camera.model.value})", view.camera, None),
        ("Equidistant 180 deg", equidistant, circle),
    ]

    backend = UTFusedBackend(view.cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * TILE_W, 2 * TILE_H))
    with torch.no_grad():
        for i in range(args.frames):
            t = args.start_s + i / args.frames * args.seconds
            tau = t / duration - 0.5 if duration > 0.0 else 0.0
            tiles = []
            for label, camera, valid in views:
                colors = backend.render(view.model, camera, tau, view.model.sh_degree).colors
                tiles.append(tile(colors if valid is None else colors * valid, label))
            writer.write(np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0))
    writer.release()
    print(f"Wrote {args.frames} frames to {args.out}")


if __name__ == "__main__":
    main()
