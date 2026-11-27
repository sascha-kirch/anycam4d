"""Interactive browser viewer for a trained checkpoint, with time playback and camera paths.

``--backend anycam4d`` renders through the ray backend and adds a lens panel (pinhole or fisheye with
k1..k4), so the scene can be viewed through a lens it was never captured with.

Usage::

    pixi run view runs/<run>/checkpoints/best.pt [--backend anycam4d] [--port 8080]
"""

from __future__ import annotations

import argparse
import time

import torch

from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.viewer import ViewerConfig
from anycam4d.enums import BackendType
from anycam4d.enums import RasterizeMode
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.viewer.factory import ViewerFactory


def main() -> None:
    """Load a checkpoint and serve the viewer until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="checkpoint written by training (best.pt or last.pt)")
    parser.add_argument("--port", type=int, default=8080, help="viewer port (default: 8080)")
    parser.add_argument("--fps", type=float, default=30.0, help="playback frame rate (default: 30)")
    parser.add_argument("--num-timesteps", type=int, default=100, help="positions on the time slider (default: 100)")
    parser.add_argument(
        "--backend",
        choices=["ewa", "anycam4d"],
        default="ewa",
        help="ewa (fast; linearizes the lens) or anycam4d (unscented transform, exact under any lens)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("The viewer renders with the CUDA kernels and needs a GPU.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FourDGaussians.from_checkpoint(checkpoint["model_state"], source=args.checkpoint).cuda().eval()
    backend = BackendType(args.backend)
    pipeline = RenderPipeline(
        backend=BackendFactory.create(RenderingConfig(backend=backend, rasterize_mode=RasterizeMode.CLASSIC))
    )

    # The model lives in the normalized scene; viewer cameras are in world space.
    T_denorm = torch.eye(4)
    T_denorm[:3, :3] *= float(checkpoint["norm_scale"])
    T_denorm[:3, 3] = torch.tensor(checkpoint["norm_center"], dtype=torch.float32)

    viewer = ViewerFactory.create(
        ViewerConfig(port=args.port, playback_fps=args.fps),
        model,
        pipeline,
        num_timesteps=args.num_timesteps,
        T_denorm=T_denorm,
    )
    print(f"{len(model):,} Gaussians, {args.backend} backend; viewer at http://localhost:{args.port}")
    while viewer is not None:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
