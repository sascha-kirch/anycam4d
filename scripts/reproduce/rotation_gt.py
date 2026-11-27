"""Rotated cameras against path-traced ground truth: PSNR by rotation θ, plus the validity gates.

Pixels count when their ray lies inside a cone around the unrotated camera's axis, so newly exposed content is excluded.

Usage: ``python scripts/reproduce/rotation_gt.py --gt DIR --scene DIR --runs DIR --output out.json [--seed 1]``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import torch


sys.path.insert(0, str(Path(__file__).parent))

from rotation_consistency import RotationProbe
from rotation_consistency import project_rays
from rotation_consistency import rotation_toward
from rotation_consistency import sample_at

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.eval.bands import incidence_degrees
from anycam4d.eval.checkpoint import denorm_transform
from anycam4d.math.camera_rays import camera_rays


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.data.types import TrainingSample

PRIMARY_CONE_DEG = 40.0
SECONDARY_CONE_DEG = 85.0
# Model label -> (run name, dataset under the scene dir, render variants): own renderer first.
MODELS = {
    "ut_nat": ("ut_nat_dykb_{seed}", "fisheye_kb/count_32", ("ut", "ewa")),
    "ewa_nat": ("ewa_nat_dykb_{seed}", "fisheye_kb/count_32", ("ewa", "ut")),
    "ut_ud": ("ut_ud_dykb_{seed}", "fisheye_kb_pinhole/count_32", ("ut",)),
}
NATIVE = "fisheye_kb/count_32"


def load_image(path: Path, device: torch.device) -> Float[Tensor, "H W 3"]:
    """JPEG to RGB in [0, 1]."""
    return torch.from_numpy(cv2.imread(str(path))[..., ::-1].copy()).float().to(device) / 255.0


def psnr(sse: float, count: int) -> float | None:
    """Pooled PSNR, or None without pixels."""
    return -10.0 * math.log10(max(sse / count, 1e-12)) if count else None


def add(table: dict[str, list[float]], key: str, err: Float[Tensor, "N"]) -> None:
    """Add per-pixel squared errors to a pooled entry."""
    entry = table.setdefault(key, [0.0, 0])
    entry[0] += float(err.sum())
    entry[1] += int(err.numel())


class Geometry:
    """Rays and masks of the native eval camera."""

    def __init__(self, sample: TrainingSample) -> None:
        """Rays and observed pixels of the sample's lens."""
        self.height, self.width = sample.image.shape[:2]
        self.rays, valid = camera_rays(sample.K, sample.distortion_params, sample.camera_model, self.width, self.height)
        self.valid: Bool[Tensor, "H W"] = valid if sample.validity is None else valid & sample.validity[..., 0]
        self.K = sample.K.double().numpy()
        self.dist = sample.distortion_params.double().numpy()
        self.family = CAMERA_REGISTRY[sample.camera_model].family

    def masks(
        self, rotation: Float[Tensor, "3 3"]
    ) -> tuple[Float[Tensor, "H W 3"], Bool[Tensor, "H W"], Bool[Tensor, "H W"]]:
        """Each rotated pixel's ray in the unrotated camera frame, with the primary and secondary masks."""
        ref_dirs = (self.rays.double().reshape(-1, 3) @ rotation).reshape(self.height, self.width, 3)
        ref_inc = incidence_degrees(ref_dirs.float())
        return ref_dirs, self.valid & (ref_inc <= PRIMARY_CONE_DEG), self.valid & (ref_inc <= SECONDARY_CONE_DEG)


def index_samples(dataset: MultiViewSequenceDataset, frames: list[int]) -> dict[tuple[int, int], int]:
    """Flat sample index per (raw camera id, frame id)."""
    return {
        (dataset.raw_camera_id(dataset[i].camera_id), dataset.frame_id_for(i)): i
        for i in range(len(dataset))
        if dataset.frame_id_for(i) in frames
    }


def score_gates(gt_dir: Path, scene: Path, index: dict, gates: dict[str, list[float]]) -> None:
    """G1: θ = 0 ground truth against the dataset image. G2: rotated against θ = 0 ground truth along identical rays."""
    device = torch.device("cuda")
    dataset = MultiViewSequenceDataset(scene / NATIVE, split="eval")
    lookup = index_samples(dataset, index["frames"])
    geometry = None
    unrotated = {c["cam"]: c["id"] for c in index["cameras"] if c["theta"] == 0}
    for cam in index["cameras"]:
        rotation = rotation_toward(cam["theta"], cam["azimuth"])
        for frame in index["frames"]:
            sample = dataset[lookup[(cam["cam"], frame)]]
            geometry = geometry or Geometry(sample)
            gt = load_image(gt_dir / "images" / f"{cam['id']}_{frame:04d}.jpg", device)
            if cam["theta"] == 0:
                err = (gt - sample.image.to(device)).pow(2).mean(dim=-1)[geometry.valid.to(device)]
                add(gates, f"G1/{cam['cam']}", err)
                continue
            ref_dirs, primary, _ = geometry.masks(rotation)
            base = load_image(gt_dir / "images" / f"{unrotated[cam['cam']]}_{frame:04d}.jpg", device)
            pixels = project_rays(ref_dirs[primary].numpy(), geometry.K, geometry.dist, geometry.family)
            add(
                gates,
                f"G2/{cam['theta']}/{cam['azimuth']}",
                (gt[primary.to(device)] - sample_at(base, pixels)).pow(2).mean(dim=-1),
            )


def score_model(label: str, seed: int, args: argparse.Namespace, index: dict, table: dict, gates: dict) -> None:
    """One model through its render variants on every rotated camera; G3 at θ = 0."""
    run, data, variants = MODELS[label]
    checkpoint_path = args.runs / run.format(seed=seed) / "checkpoints" / "last.pt"
    probe = RotationProbe(checkpoint_path, args.scene / data, 1)
    denorm = denorm_transform(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    native = MultiViewSequenceDataset(args.scene / NATIVE, split="eval")  # the undistorted model is drawn natively too
    native.apply_viewmat_transform(denorm)
    lookup = index_samples(native, index["frames"])
    geometry = None
    for cam in index["cameras"]:
        rotation = rotation_toward(cam["theta"], cam["azimuth"])
        for frame in index["frames"]:
            sample = native[lookup[(cam["cam"], frame)]]
            geometry = geometry or Geometry(sample)
            _, primary, secondary = geometry.masks(rotation)
            primary, secondary = primary.to(probe.device), secondary.to(probe.device)
            gt = load_image(args.gt / "images" / f"{cam['id']}_{frame:04d}.jpg", probe.device)
            camera = probe.camera(sample, rotation, 1)
            for variant in variants:
                render = probe.render(variant, camera, sample.timestamp)
                err = (render - gt).pow(2).mean(dim=-1)
                key = f"{label}/{variant}/{seed}/{cam['theta']}"
                add(table, f"{key}/primary", err[primary])
                add(table, f"{key}/secondary", err[secondary])
                if cam["theta"] == 0:
                    add(gates, f"G3/{label}/{variant}/{seed}/new_gt", err[secondary])
                    dataset_err = (render - sample.image.to(probe.device)).pow(2).mean(dim=-1)
                    add(gates, f"G3/{label}/{variant}/{seed}/dataset", dataset_err[secondary])


def main() -> None:
    """Gates, then every model; writes pooled squared errors so θ ranges can be pooled afterwards."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gt", type=Path, required=True, help="Directory with images/ and index.json.")
    parser.add_argument("--scene", type=Path, required=True, help="Dynopsis scene1 directory.")
    parser.add_argument("--runs", type=Path, required=True, help="Directory of the trained runs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    index = json.loads((args.gt / "index.json").read_text())
    table: dict[str, list[float]] = {}
    gates: dict[str, list[float]] = {}
    score_gates(args.gt, args.scene, index, gates)
    for label in MODELS:
        score_model(label, args.seed, args, index, table, gates)
        torch.cuda.empty_cache()
    result = {
        "seed": args.seed,
        "frames": index["frames"],
        "gates": {k: {"psnr": psnr(*v), "px": v[1]} for k, v in gates.items()},
        "psnr": {k: {"psnr": psnr(*v), "sse": v[0], "px": v[1]} for k, v in table.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    for section in ("gates", "psnr"):
        for key, value in sorted(result[section].items()):
            print(f"{key:40s} {value['psnr']:6.2f} ({value['px']} px)")


if __name__ == "__main__":
    main()
