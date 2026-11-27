"""Render a video of one checkpoint whose lens morphs from pinhole to fisheye and back while the scene plays.

The lens is Kannala-Brandt throughout: ``k = (1 - s) · k_tan`` blends tan θ's Taylor series (≈ pinhole, 0.1 % radial
error at 45°) into equidistant ``r = f θ`` (s = 1) while the diagonal field of view goes ``--fov-start`` → ``--fov-end``.
The diagonal is capped at 180°: geometry behind the camera plane is culled, so pixels past 90° stay empty.
Camera: eval sample ``--index``. ``--trained-renderer`` draws an EWA checkpoint with EWA through each morph lens.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.enums import CameraModelType
from anycam4d.eval.scene_view import load_eval_view
from anycam4d.eval.scene_view import to_bgr8
from anycam4d.renderer.ut_fused_backend import UTFusedBackend


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


# tan θ = θ (1 + θ²/3 + 2θ⁴/15 + 17θ⁶/315 + 62θ⁸/2835 + …): the Kannala-Brandt coefficients of a pinhole.
K_TAN = (1.0 / 3.0, 2.0 / 15.0, 17.0 / 315.0, 62.0 / 2835.0)


def morph_camera(
    s: float, fov_start: float, fov_end: float, viewmat: Float[Tensor, "4 4"], width: int, height: int
) -> BatchedCamera:
    """Kannala-Brandt camera at morph state ``s`` ∈ [0, 1] whose image diagonal spans the interpolated FOV."""
    half_fov = math.radians(fov_start + s * (fov_end - fov_start)) / 2
    k = [(1.0 - s) * c for c in K_TAN]
    theta_d = half_fov * (1 + k[0] * half_fov**2 + k[1] * half_fov**4 + k[2] * half_fov**6 + k[3] * half_fov**8)
    focal = math.hypot(width, height) / 2 / theta_d
    device = viewmat.device
    K = torch.tensor([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]], device=device)
    return BatchedCamera(
        viewmats=viewmat[None],
        Ks=K[None],
        model=CameraModelType.FISHEYE_OPENCV,
        width=width,
        height=height,
        distortion=torch.tensor([k], device=device),
    )


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("data_dir")
    parser.add_argument("out", type=Path, help="Output .mp4")
    parser.add_argument("--index", type=int, default=0, help="Eval-split sample giving the pose.")
    parser.add_argument("--fov-start", type=float, default=90.0, help="Diagonal FOV of the pinhole end, degrees.")
    parser.add_argument("--fov-end", type=float, default=180.0, help="Diagonal FOV of the fisheye end, degrees.")
    parser.add_argument("--width", type=int, default=None, help="Default: the dataset camera's width.")
    parser.add_argument("--height", type=int, default=None, help="Default: the dataset camera's height.")
    parser.add_argument("--start-s", type=float, default=0.0, help="Capture second the window starts at.")
    parser.add_argument("--seconds", type=float, default=4.0, help="Capture seconds rendered (0: frozen time).")
    parser.add_argument("--frames", type=int, default=120, help="One full pinhole → fisheye → pinhole cycle.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-label", action="store_true", help="Omit the FOV overlay.")
    parser.add_argument(
        "--trained-renderer",
        action="store_true",
        help="Draw with the renderer the checkpoint was trained with (EWA for an EWA checkpoint) instead of AnyCam4D.",
    )
    args = parser.parse_args()
    if max(args.fov_start, args.fov_end) > 180.0:
        parser.error("diagonal FOV past 180°: geometry behind the camera plane is culled, so the rim would be empty")
    return args


def main() -> None:
    """Write the morph video."""
    args = parse_args()

    view = load_eval_view(args.checkpoint, args.data_dir, args.index)
    model, viewmat = view.model, view.camera.viewmats[0]
    width, height = args.width or view.camera.width, args.height or view.camera.height
    duration = view.dataset.t_max - view.dataset.t_min
    if duration > 0.0 and args.start_s + args.seconds > duration:
        raise SystemExit(f"--start-s + --seconds exceeds the {duration:.2f} s capture; the model is untrained there")

    backend = view.trained_backend if args.trained_renderer else UTFusedBackend(view.cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    with torch.no_grad():
        for i in range(args.frames):
            s = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / args.frames)  # eased ping-pong, loops seamlessly
            t = args.start_s + i / args.frames * args.seconds  # capture seconds since t_min
            tau = t / duration - 0.5 if duration > 0.0 else 0.0
            camera = morph_camera(s, args.fov_start, args.fov_end, viewmat, width, height)
            colors = backend.render(model, camera, tau, model.sh_degree).colors
            fov = args.fov_start + s * (args.fov_end - args.fov_start)
            writer.write(to_bgr8(colors, "" if args.no_label else f"diagonal FOV {fov:.0f} deg"))
    writer.release()
    print(f"Wrote {args.frames} frames to {args.out}")


if __name__ == "__main__":
    main()
