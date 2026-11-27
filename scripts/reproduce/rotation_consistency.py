"""Rotation consistency: does a renderer draw content alike as it moves through the field?

A camera rotated about its own centre has no parallax and leaves every camera-to-Gaussian direction, colour and time
unchanged, so an exact projection draws the same content at the new incidence. The eval camera is rotated by θ toward
several image azimuths; each rotated pixel is compared, along the identical ray, with a 2x reference render at θ = 0
restricted to its central cone. Every model is rendered by UT, EWA and EWA without dilation; each variant is scored
against its own reference and against UT's. The result is pooled per incidence band of the rotated pixel.

Usage: ``python scripts/reproduce/rotation_consistency.py <checkpoint.pt> --output out.json [--data_dir DIR]
[--tb_dir DIR] [--figure DIR]``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.camera.registry import ProjectionFamily
from anycam4d.config.dataset import DatasetConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.eval.bands import incidence_degrees
from anycam4d.eval.checkpoint import build_pipeline_from_checkpoint
from anycam4d.eval.checkpoint import denorm_transform
from anycam4d.math.camera_rays import camera_rays
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.data.types import TrainingSample
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

# Render variant -> (backend, eps2d factor). ewa_nodil is diagnostic: EWA without its screen-space dilation, which
# separates the dilation (fixed in pixels, so incidence-dependent in angle) from the linearization.
VARIANTS = {"ut": ("anycam4d", 1.0), "ewa": ("ewa", 1.0), "ewa_nodil": ("ewa", 0.0)}
THETAS_DEG = (5, 10, 20, 30, 40, 50, 60, 70, 80)
AZIMUTHS_DEG = (0, 90, 180, 270)
BAND_EDGES_DEG = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
FLOOR_THETA_DEG = 5
FIGURE_THETAS_DEG = (0, 40, 70)


def rotation_toward(theta_deg: float, azimuth_deg: float) -> Float[Tensor, "3 3"]:
    """Camera-frame rotation that carries the optical axis to incidence ``theta`` toward image azimuth ``azimuth``."""
    theta, phi = math.radians(theta_deg), math.radians(azimuth_deg)
    ax, ay = -math.sin(phi), math.cos(phi)
    skew = torch.tensor([[0.0, 0.0, ay], [0.0, 0.0, -ax], [-ay, ax, 0.0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * skew @ skew


def project_rays(
    dirs: Float[np.ndarray, "N 3"], K: Float[np.ndarray, "3 3"], dist: Float[np.ndarray, "P"], family: ProjectionFamily
) -> Float[np.ndarray, "N 2"]:
    """Camera-space rays to continuous pixel coordinates (``i + 0.5`` = pixel ``i``); inverse of ``camera_rays``."""
    obj = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1, 1, 3)
    zero = np.zeros((3, 1), dtype=np.float64)
    if family is ProjectionFamily.FISHEYE_FAMILY:
        pix, _ = cv2.fisheye.projectPoints(obj, zero, zero, K, dist)
    else:
        pix, _ = cv2.projectPoints(obj, zero, zero, K, dist)
    return pix.reshape(-1, 2)


def sample_at(image: Float[Tensor, "H W 3"], pixels: Float[np.ndarray, "N 2"]) -> Float[Tensor, "N 3"]:
    """Bilinear lookup of ``image`` at continuous pixel coordinates (``align_corners=False`` uses the same convention)."""
    height, width = image.shape[:2]
    grid = torch.from_numpy(pixels).to(image.device, torch.float32)
    grid = torch.stack([2.0 * grid[:, 0] / width - 1.0, 2.0 * grid[:, 1] / height - 1.0], dim=-1)
    out = F.grid_sample(image.permute(2, 0, 1)[None], grid[None, None], mode="bilinear", align_corners=False)
    return out[0, :, 0].T


class RotationProbe:
    """One model, its two pipelines and the eval camera geometry."""

    def __init__(self, checkpoint_path: Path, data_dir: Path | None, supersample: int) -> None:
        """Load the model once and build a pipeline per render variant and resolution."""
        self.device = torch.device("cuda")
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        cfg = checkpoint["config"]
        self.trained_backend = cfg["rendering"]["backend"]
        model, _ = build_pipeline_from_checkpoint(checkpoint, self.device, source=str(checkpoint_path))
        self.model: FourDGaussians = model.eval()
        # eps2d is a dilation in pixels^2 (EWA blurs the image with it, UT only its tile box): a reference at
        # sx resolution gets s^2 * eps2d, the same angular dilation as the 1x render it is compared with.
        self.pipelines: dict[tuple[str, int], RenderPipeline] = {}
        for variant, (backend, eps_factor) in VARIANTS.items():
            for scale in {1, supersample}:
                eps2d = max(cfg["rendering"]["eps2d"] * eps_factor * scale**2, 1e-6)
                override = {"backend": backend, "eps2d": eps2d}
                render_cfg = RenderingConfig.from_omegaconf(OmegaConf.merge(cfg["rendering"], override))
                self.pipelines[(variant, scale)] = RenderPipeline(backend=BackendFactory.create(render_cfg))
        data_dir = data_dir or DatasetConfig.from_omegaconf(OmegaConf.create(cfg["dataset"])).data_dir
        self.dataset = MultiViewSequenceDataset(data_dir, split="eval")
        t_denorm = denorm_transform(checkpoint)
        if t_denorm is not None:
            self.dataset.apply_viewmat_transform(t_denorm)

    def camera(self, sample: TrainingSample, rotation: Float[Tensor, "3 3"], scale: int) -> BatchedCamera:
        """The sample's camera rotated about its centre by ``rotation`` (camera frame), at ``scale``x resolution."""
        viewmat = sample.viewmat.to(self.device).double().clone()
        rot = rotation.to(self.device)
        viewmat[:3, :] = rot @ viewmat[:3, :]
        K = sample.K.to(self.device).clone()
        K[:2] *= scale
        height, width = sample.image.shape[:2]
        return BatchedCamera(
            viewmats=viewmat.float()[None],
            Ks=K[None],
            model=sample.camera_model,
            width=width * scale,
            height=height * scale,
            distortion=sample.distortion_params.to(self.device)[None],
        )

    @torch.no_grad()
    def render(self, variant: str, camera: BatchedCamera, timestamp: float, scale: int = 1) -> Float[Tensor, "H W 3"]:
        """Clamped RGB render: every reported metric scores the render clamped to [0, 1]."""
        out = self.pipelines[(variant, scale)].render(self.model, camera, timestamp, self.model.sh_degree)
        return out.colors[0].clamp(0.0, 1.0)


class Accumulator:
    """Squared error and sample count per (backend, reference, θ, band)."""

    def __init__(self) -> None:
        """Start empty."""
        self.sums: dict[tuple[str, str, int, int], list[float]] = {}

    def add(self, key: tuple[str, str, int], err: Float[Tensor, "N"], band: Float[Tensor, "N"]) -> None:
        """Add per-pixel squared errors ``err`` (channel mean) binned by ``band`` index."""
        for b in torch.unique(band).tolist():
            sel = band == b
            entry = self.sums.setdefault((*key, int(b)), [0.0, 0])
            entry[0] += float(err[sel].sum())
            entry[1] += int(sel.sum())

    def psnr(self, backend: str, ref: str, thetas: list[int], band: int | None) -> tuple[float | None, int]:
        """Pooled PSNR over ``thetas`` (and one ``band``, or all), with the pixel count."""
        sse = n = 0
        for (b, r, t, k), (s, c) in self.sums.items():
            if b == backend and r == ref and t in thetas and (band is None or k == band):
                sse, n = sse + s, n + c
        if n == 0:
            return None, 0
        return -10.0 * math.log10(max(sse / n, 1e-12)), n


def compare(
    probe: RotationProbe,
    sample: TrainingSample,
    refs: dict[str, Float[Tensor, "H W 3"]],
    acc: Accumulator,
    cone_deg: float,
    supersample: int,
) -> None:
    """Score every variant at every (θ, azimuth) against its own reference and UT's, for one eval sample."""
    height, width = sample.image.shape[:2]
    family = CAMERA_REGISTRY[sample.camera_model].family
    rays, valid = camera_rays(sample.K, sample.distortion_params, sample.camera_model, width, height)
    if sample.validity is not None:
        valid &= sample.validity[..., 0]
    K_ref = sample.K.double().numpy().copy()
    K_ref[:2] *= supersample
    dist = sample.distortion_params.double().numpy()
    band_all = (incidence_degrees(rays) // 10).clamp(0, len(BAND_EDGES_DEG) - 2)
    for theta in THETAS_DEG:
        for azimuth in AZIMUTHS_DEG:
            rotation = rotation_toward(theta, azimuth)
            ref_dirs = rays.double().reshape(-1, 3) @ rotation  # R^T d for row vectors
            ref_inc = incidence_degrees(ref_dirs.float().reshape(height, width, 3))
            keep: Bool[Tensor, "H W"] = valid & (ref_inc <= cone_deg)
            if not bool(keep.any()):
                continue
            pixels = project_rays(ref_dirs.reshape(height, width, 3)[keep].numpy(), K_ref, dist, family)
            band = band_all[keep].to(probe.device)
            camera = probe.camera(sample, rotation, 1)
            for variant in VARIANTS:
                test = probe.render(variant, camera, sample.timestamp)[keep.to(probe.device)]
                for ref_kind, target in (("self", refs[variant]), ("ut", refs["ut"])):
                    err = (test - sample_at(target, pixels)).pow(2).mean(dim=-1)
                    acc.add((variant, ref_kind, theta), err, band)


def figure(probe: RotationProbe, sample: TrainingSample, out_dir: Path, cone_deg: float) -> None:
    """Same content seen centrally and at 40°/70°, resampled into the reference layout: rows = backends."""
    height, width = sample.image.shape[:2]
    family = CAMERA_REGISTRY[sample.camera_model].family
    rays, valid = camera_rays(sample.K, sample.distortion_params, sample.camera_model, width, height)
    cone = valid & (incidence_degrees(rays) <= cone_deg)
    rows, cols = torch.nonzero(cone, as_tuple=True)
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    K = sample.K.double().numpy()
    dist = sample.distortion_params.double().numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = []
    for backend in ("ut", "ewa"):
        tiles = []
        for theta in FIGURE_THETAS_DEG:
            rotation = rotation_toward(theta, 0.0)
            test = probe.render(backend, probe.camera(sample, rotation, 1), sample.timestamp)
            dirs = rays.double()[r0:r1, c0:c1].reshape(-1, 3) @ rotation.T  # R d for row vectors
            tile = sample_at(test, project_rays(dirs.numpy(), K, dist, family)).reshape(r1 - r0, c1 - c0, 3)
            tile = tile * cone[r0:r1, c0:c1, None].to(tile.device)
            tile_u8 = cv2.cvtColor((tile.cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.putText(tile_u8, f"{backend} {theta} deg", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(str(out_dir / f"{backend}_theta{theta:02d}.png"), tile_u8)
            tiles.append(tile_u8)
        grid.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(out_dir / "grid.png"), np.concatenate(grid, axis=0))


def summarize(acc: Accumulator) -> dict:
    """Per variant and reference: PSNR per θ, per band (θ ≥ 10 pooled), the θ = 5 floor and the drop per band."""
    bands = range(len(BAND_EDGES_DEG) - 1)
    rotated = [t for t in THETAS_DEG if t != FLOOR_THETA_DEG]
    out: dict = {}
    for backend in VARIANTS:
        for ref in ("self", "ut"):
            floor, _ = acc.psnr(backend, ref, [FLOOR_THETA_DEG], None)
            per_band = {}
            for b in bands:
                psnr, n = acc.psnr(backend, ref, rotated, b)
                if n:
                    label = f"{BAND_EDGES_DEG[b]}-{BAND_EDGES_DEG[b + 1]}"
                    per_band[label] = {"psnr": psnr, "drop": psnr - floor if floor is not None else None, "px": n}
            per_theta = {t: acc.psnr(backend, ref, [t], None)[0] for t in THETAS_DEG}
            out[f"{backend}/{ref}"] = {"floor": floor, "bands": per_band, "theta": per_theta}
    return out


def write_tb(summary: dict, tb_dir: Path) -> None:
    """Band PSNR and drop per backend/reference to TensorBoard, stepped by band index."""
    writer = SummaryWriter(str(tb_dir))
    for key, entry in summary.items():
        tag = "rotation/" + key.replace("/", "_ref_")
        for label, values in entry["bands"].items():
            index = int(label.split("-")[0]) // 10
            writer.add_scalar(f"{tag}/band_psnr", values["psnr"], index)
            if values["drop"] is not None:
                writer.add_scalar(f"{tag}/band_drop", values["drop"], index)
        if entry["floor"] is not None:
            writer.add_scalar(f"{tag}/floor", entry["floor"], 0)
    writer.close()


def main() -> None:
    """Run the probe on evenly spaced eval samples and write JSON (+ TB, + figure)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--tb_dir", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None, help="Write the qualitative strip for the first sample.")
    parser.add_argument("--samples", type=int, default=5, help="Evenly spaced eval samples (timestamps).")
    parser.add_argument("--cone", type=float, default=20.0, help="Reference cone half-angle, degrees.")
    parser.add_argument("--supersample", type=int, default=2, help="Reference resolution factor.")
    args = parser.parse_args()

    probe = RotationProbe(args.checkpoint, args.data_dir, args.supersample)
    indices = np.linspace(0, len(probe.dataset) - 1, args.samples).round().astype(int).tolist()
    acc = Accumulator()
    for index in indices:
        sample = probe.dataset[index]
        ref_camera = probe.camera(sample, torch.eye(3, dtype=torch.float64), args.supersample)
        refs = {v: probe.render(v, ref_camera, sample.timestamp, args.supersample) for v in VARIANTS}
        compare(probe, sample, refs, acc, args.cone, args.supersample)
    if args.figure is not None:
        figure(probe, probe.dataset[indices[0]], args.figure, args.cone)

    summary = summarize(acc)
    result = {
        "checkpoint": str(args.checkpoint),
        "trained_backend": probe.trained_backend,
        "camera_model": probe.dataset.camera_model.value,
        "samples": indices,
        "cone_deg": args.cone,
        "supersample": args.supersample,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    if args.tb_dir is not None:
        write_tb(summary, args.tb_dir)
    for key, entry in summary.items():
        drops = " ".join(f"{k}:{v['drop']:+.2f}" for k, v in entry["bands"].items() if v["drop"] is not None)
        print(f"{key:9s} floor {entry['floor']:.2f}  drop {drops}")


if __name__ == "__main__":
    main()
