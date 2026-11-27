"""Render thin-lens depth of field from one checkpoint through the dataset's own lens.

Uses ``anycam4d.renderer.depth_of_field``: ``--samples 1`` is the single blurred pass, ``K`` averages K partly blurred
passes, which is more accurate on high-contrast edges.
Focus depth: ``--focus`` (camera depth, normalized scene units), else autofocus on the opaque front surface under
``--focus-px``. ``--blur-px`` sets the aperture by the blur radius of a point at infinity. A ``.png`` output writes
sharp | depth of field side by side at eval sample ``--index``; an ``.mp4`` output writes a rack focus from ``--focus``
to ``--focus-to`` (default: the near to the far surface in view) at a fixed aperture, frozen at that sample or, with
``--seconds``, while the scene plays (focus there and back).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.eval.scene_view import load_eval_view
from anycam4d.eval.scene_view import to_bgr8
from anycam4d.math.lens_domain import unproject_pixels
from anycam4d.primitives.fourdgaussians.conditioning import condition_at_t
from anycam4d.renderer.depth_of_field import render_depth_of_field
from anycam4d.renderer.depth_of_field import render_thin_lens


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians


def surface_depths(
    model: FourDGaussians, tau: float, camera: BatchedCamera, pixels: Float[np.ndarray, "P 2"]
) -> list[float | None]:
    """Front-surface depth under each pixel: 20th-percentile camera z of opaque means within 1° of its ray.

    ``None`` where the pixel is outside the lens, looks past 90°, or has no opaque Gaussian on its ray.
    """
    rays, valid = unproject_pixels(
        pixels,
        camera.Ks[0].double().cpu().numpy(),
        camera.distortion[0].double().cpu().numpy(),
        CAMERA_REGISTRY[camera.model].family,
    )
    viewmat = camera.viewmats[0]
    with torch.no_grad():
        cond = condition_at_t(model, tau, prefilter_threshold=model.prefilter_threshold)
        opaque = cond.opacity_cond > 0.5
        cam = cond.means_3d_cond[opaque] @ viewmat[:3, :3].T + viewmat[:3, 3]
        dirs = cam / cam.norm(dim=1, keepdim=True).clamp_min(1e-6)
    depths: list[float | None] = []
    for ray, ok in zip(rays, valid, strict=True):
        keep = dirs @ torch.tensor(ray, device=cam.device, dtype=torch.float32) > math.cos(math.radians(1.0))
        usable = ok and ray[2] > 0.0 and bool(keep.any())
        depths.append(float(torch.quantile(cam[keep, 2], 0.2)) if usable else None)
    return depths


def depth_range(model: FourDGaussians, tau: float, camera: BatchedCamera) -> tuple[float, float]:
    """Near and far surface in view: 10th and 90th percentile of the surface depths on a pixel grid."""
    xs = (np.arange(8) + 0.5) / 8 * camera.width
    ys = (np.arange(6) + 0.5) / 6 * camera.height
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    found = [d for d in surface_depths(model, tau, camera, grid) if d is not None]
    if len(found) < 2:
        raise SystemExit("rack focus: too few opaque surfaces in view; pass --focus and --focus-to")
    near, far = np.quantile(found, [0.1, 0.9])
    return float(near), float(far)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("data_dir")
    parser.add_argument("out", type=Path, help="Output .png (still) or .mp4 (rack focus)")
    parser.add_argument("--index", type=int, default=0, help="Eval-split sample giving the pose and time.")
    parser.add_argument("--focus", type=float, default=None, help="Focus depth, normalized scene units.")
    parser.add_argument("--focus-px", type=float, nargs=2, default=None, help="Autofocus pixel; default centre.")
    parser.add_argument(
        "--focus-to", type=float, default=None, help="Rack-focus end depth (.mp4 only); default the far surface."
    )
    parser.add_argument("--blur-px", type=float, default=30.0, help="Blur radius of a point at infinity, pixels.")
    parser.add_argument("--samples", type=int, default=8, help="Partly blurred passes per frame: 1 or >= 3.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Capture seconds played from --index (0: frozen).")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-label", action="store_true", help="Omit the overlays.")
    args = parser.parse_args()
    if args.out.suffix != ".mp4" and args.focus_to is not None:
        parser.error("--focus-to goes with an .mp4 output only")
    if args.samples < 1 or args.samples == 2:
        parser.error("--samples must be 1 or >= 3")
    if args.blur_px < 0.0:
        parser.error("--blur-px must be >= 0")
    return args


def main() -> None:
    """Write the still or the rack-focus video."""
    args = parse_args()
    view = load_eval_view(args.checkpoint, args.data_dir, args.index)
    model, cfg, camera = view.model, view.cfg, view.camera
    tau, f, sh = view.sample.timestamp, float(view.sample.K[0, 0]), model.sh_degree

    focus, focus_to = args.focus, args.focus_to
    if args.out.suffix == ".mp4":
        near, far = depth_range(model, tau, camera)
        focus = near if focus is None else focus
        focus_to = far if focus_to is None else focus_to
    elif focus is None:
        pixel = args.focus_px if args.focus_px else (camera.width / 2, camera.height / 2)
        focus = surface_depths(model, tau, camera, np.array([pixel]))[0]
        if focus is None:
            raise SystemExit(f"autofocus: no opaque surface under pixel {pixel}; pass --focus or move --focus-px")
    aperture = args.blur_px * focus / f  # blur radius at infinity = f · aperture / focus
    span = "" if focus_to is None else f" to {focus_to:.4g}"
    print(f"focus {focus:.4g}{span}, aperture radius {aperture:.4g}, {args.samples} pass(es) per frame")

    def defocus(depth: float, at: float = tau) -> Float[Tensor, "H W 3"]:
        return render_depth_of_field(model, camera, at, sh, cfg, focus=depth, aperture=aperture, samples=args.samples)[
            0
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if focus_to is None:
            sharp = render_thin_lens(model, camera, tau, sh, cfg, focus=focus)[0]
            label = "" if args.no_label else f"focus {focus:.3g}"
            cv2.imwrite(str(args.out), np.concatenate([to_bgr8(sharp), to_bgr8(defocus(focus), label)], axis=1))
            print(f"Wrote {args.out}")
            return
        duration = view.dataset.t_max - view.dataset.t_min
        if tau + args.seconds / duration > 0.5:
            raise SystemExit(f"--index + --seconds exceeds the {duration:.2f} s capture; the model is untrained there")
        size = (camera.width, camera.height)
        writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
        for i in range(args.frames):
            cycles = 2.0 if args.seconds > 0.0 else 1.0  # playing: there and back, so the clip loops
            s = 0.5 - 0.5 * math.cos(cycles * math.pi * i / max(args.frames - 1, 1))
            depth = 1.0 / ((1.0 - s) / focus + s / focus_to)  # even in 1/depth, so blur changes evenly
            at = tau + i / args.frames * args.seconds / duration
            writer.write(to_bgr8(defocus(depth, at), "" if args.no_label else f"focus {depth:.3g}"))
        writer.release()
    print(f"Wrote {args.frames} frames to {args.out}")


if __name__ == "__main__":
    main()
