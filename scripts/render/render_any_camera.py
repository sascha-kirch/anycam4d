"""Render one trained model through several camera models from the same pose and time.

No training: the model is fixed, only the camera changes. Views: the dataset's own camera (with its validity mask), a
90° pinhole and a 180° equidistant fisheye (its image circle), square at the native image height. Nothing past 180°:
geometry behind the camera plane is culled, so a wider lens would show an empty ring.
Usage: ``python scripts/render/render_any_camera.py <checkpoint.pt> <data_dir> <out_dir> [--index 0]``.
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
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.enums import CameraModelType
from anycam4d.eval.checkpoint import build_pipeline_from_checkpoint
from anycam4d.eval.checkpoint import denorm_transform


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from jaxtyping import UInt8
    from torch import Tensor


def square_camera(
    model: CameraModelType, focal: float, size: int, viewmat: Float[Tensor, "4 4"], device: torch.device
) -> BatchedCamera:
    """Square, centred, distortion-free camera of ``model`` at ``viewmat``."""
    K = torch.tensor([[focal, 0.0, size / 2], [0.0, focal, size / 2], [0.0, 0.0, 1.0]], device=device)
    return BatchedCamera(
        viewmats=viewmat[None],
        Ks=K[None],
        model=model,
        width=size,
        height=size,
        distortion=torch.zeros(1, distortion_width(model), device=device),
    )


def validity_of(camera: BatchedCamera, max_incidence_deg: float | None) -> Bool[Tensor, "H W 1"]:
    """``[H, W, 1]`` lens footprint of ``camera``, bounded by ``max_incidence_deg``."""
    mask = compute_validity(
        camera.model,
        camera.Ks[0],
        camera.distortion[0],
        camera.width,
        camera.height,
        max_incidence_deg=max_incidence_deg,
    )
    return mask.to(camera.Ks.device)[..., None]


def to_uint8(colors: Float[Tensor, "1 H W 3"], valid: Bool[Tensor, "H W 1"]) -> UInt8[np.ndarray, "H W 3"]:
    """``[1, H, W, 3]`` render, black outside ``valid`` → BGR uint8 for OpenCV."""
    rgb = (colors[0].clamp(0, 1) * valid * 255).round().byte().cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    """Render the views and write one PNG each plus a labelled mosaic."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("data_dir")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--index", type=int, default=0, help="Eval-split sample giving the pose and time.")
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model, pipeline = build_pipeline_from_checkpoint(checkpoint, device, source=args.checkpoint)
    model.eval()
    dataset = MultiViewSequenceDataset(args.data_dir, split="eval")
    t_denorm = denorm_transform(checkpoint)
    if t_denorm is not None:
        dataset.apply_viewmat_transform(t_denorm)
    sample = dataset[args.index]
    viewmat = sample.viewmat.to(device)
    size = sample.image.shape[0]

    native = BatchedCamera(
        viewmats=viewmat[None],
        Ks=sample.K.to(device)[None],
        model=sample.camera_model,
        width=sample.image.shape[1],
        height=size,
        distortion=sample.distortion_params.to(device)[None],
    )
    native_valid = torch.ones(size, native.width, 1, device=device)
    if sample.validity is not None:
        native_valid = sample.validity.to(device).float()
    pinhole = square_camera(CameraModelType.PINHOLE, size / 2, size, viewmat, device)
    fisheye = square_camera(CameraModelType.FISHEYE_EQUIDISTANT, size / math.pi, size, viewmat, device)
    views = [
        ("native", f"native ({sample.camera_model.value})", native, native_valid),
        ("pinhole90", "pinhole 90 deg", pinhole, torch.ones(size, size, 1, device=device)),
        ("fisheye180", "fisheye 180 deg", fisheye, validity_of(fisheye, 90.0).float()),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    with torch.no_grad():
        for name, label, camera, valid in views:
            image = to_uint8(pipeline.render(model, camera, sample.timestamp, model.sh_degree).colors, valid)
            cv2.imwrite(str(args.out_dir / f"{name}.png"), image)
            tile = cv2.resize(
                image, (round(image.shape[1] * size / image.shape[0]), size), interpolation=cv2.INTER_AREA
            )
            cv2.putText(tile, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
    cv2.imwrite(str(args.out_dir / "mosaic.png"), np.concatenate(tiles, axis=1))
    print(f"Wrote {len(tiles)} views and mosaic.png to {args.out_dir}")


if __name__ == "__main__":
    main()
