"""Native-space full/central/peripheral evaluation, and the observation footprint of a native camera."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anycam4d.camera.validity import compute_validity
from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.enums import BackendType
from anycam4d.enums import CameraModelType
from anycam4d.eval import native_evaluator as native_evaluator_module
from anycam4d.eval.native_evaluator import NativeRegionEvaluator
from anycam4d.eval.native_evaluator import _native_camera_geometry
from anycam4d.eval.metrics import masked_psnr
from anycam4d.math.camera_rays import camera_rays
from anycam4d.math.region import PinholeField
from anycam4d.math.region import central_mask_from_rays
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline


requires_cuda = pytest.mark.skipif(not HAS_CUDA_EXT, reason="render path runs on the CUDA kernels")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ut_pipeline() -> RenderPipeline:
    """UT supports every camera model (pinhole + both fisheye/opencv families)."""
    backend = BackendFactory.create(RenderingConfig(backend=BackendType.ANYCAM4D))
    return RenderPipeline(backend=backend)


def _assert_finite_metric(value: float) -> None:
    assert math.isfinite(value) or value == float("inf"), f"non-finite metric: {value}"


def _write_dataset(
    tmp_path: Path,
    *,
    model: str,
    width: int,
    height: int,
    params: list[float],
    n_cams: int = 2,
) -> Path:
    """A minimal MultiViewSequence dataset: `n_cams` cameras x 2 frames, one shared camera spec."""
    (tmp_path / "images").mkdir()
    frame_times = [0.0, 1.0 / 30.0]
    rng = np.random.default_rng(11)

    camera_spec: dict = {"model": model, "width": width, "height": height, "params": params}
    cameras = [{**camera_spec, "camera_id": cam_idx} for cam_idx in range(n_cams)]

    captures: list[dict] = []
    for frame_idx, t in enumerate(frame_times):
        for cam_idx in range(n_cams):
            rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            alpha = np.full((height, width, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([rgb, alpha], axis=2)
            name = f"cam{cam_idx:02d}_{frame_idx:04d}"
            Image.fromarray(rgba, mode="RGBA").save(tmp_path / "images" / f"{name}.png")
            captures.append(
                {
                    "image_path": f"images/{name}.png",
                    "camera_id": cam_idx,
                    "frame_id": frame_idx,
                    "timestamp": t,
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [cam_idx * 0.5, 0.0, 3.0],
                }
            )

    metadata = {
        "world_coordinate_system": "right_handed",
        "camera_coordinate_system": "opencv",
        "transform_convention": "camera_to_world",
    }
    schema = {"schema_version": 1, "metadata": metadata, "cameras": cameras, "captures": captures}
    with open(tmp_path / "train_dataset.json", "w") as f:
        json.dump(schema, f)
    return tmp_path


@pytest.fixture()
def tmp_opencv_dataset(tmp_path: Path) -> Path:
    """OPENCV native camera with k1 = -0.3: 65% observed, ~62% central / ~2.6% peripheral."""
    return _write_dataset(tmp_path, model="opencv", width=64, height=48, params=[35.0, 35.0, 31.5, 23.5, -0.3])


@pytest.fixture()
def tmp_fisheye_opencv_dataset(tmp_path: Path) -> Path:
    """FISHEYE_OPENCV native camera with k1 = -0.05: fully observed, ~22% peripheral."""
    return _write_dataset(tmp_path, model="fisheye_opencv", width=64, height=48, params=[30.0, 30.0, 31.5, 23.5, -0.05])


@requires_cuda
def test_native_eval_pinhole_degenerate(tmp_dataset: Path) -> None:
    """Without a central reference only `full` is scored."""
    device = _device()
    ds = MultiViewSequenceDataset(tmp_dataset, split="train")
    model = FourDGaussians.from_random(n_points=200, sh_degree=1).to(device).eval()

    result = NativeRegionEvaluator(device, lpips_net="alex").evaluate(model, _ut_pipeline(), ds).result

    assert "full_psnr" in result.mean
    assert "central_psnr" not in result.mean  # no reference, no split
    assert "peripheral_psnr" not in result.mean
    assert "peripheral_ssim" not in result.mean


@requires_cuda
def test_native_eval_opencv_nondegenerate(tmp_opencv_dataset: Path) -> None:
    """OPENCV native source with real radial distortion: central and peripheral both present."""
    device = _device()
    ds = MultiViewSequenceDataset(tmp_opencv_dataset, split="train")
    model = FourDGaussians.from_random(n_points=200, sh_degree=1).to(device).eval()

    field = PinholeField(35.0, 35.0, 31.5, 23.5, 64.0, 48.0)
    fields = {ds.raw_camera_id(i): field for i in range(ds.cameras_per_timestep)}
    evaluator = NativeRegionEvaluator(device, lpips_net="alex", central_fields=fields)
    result = evaluator.evaluate(model, _ut_pipeline(), ds).result

    for key in (
        "central_lpips",
        "full_psnr",
        "full_ssim",
        "central_psnr",
        "central_ssim",
        "peripheral_psnr",
        "peripheral_ssim",
        "lpips",
    ):
        assert key in result.mean, f"missing {key}"
        _assert_finite_metric(result.mean[key])


@requires_cuda
def test_native_eval_fisheye_opencv_nondegenerate(tmp_fisheye_opencv_dataset: Path) -> None:
    """FISHEYE_OPENCV native source, scored directly: all three regions present."""
    device = _device()
    ds = MultiViewSequenceDataset(tmp_fisheye_opencv_dataset, split="train")
    model = FourDGaussians.from_random(n_points=200, sh_degree=1).to(device).eval()

    field = PinholeField(35.0, 35.0, 31.5, 23.5, 64.0, 48.0)
    fields = {ds.raw_camera_id(i): field for i in range(ds.cameras_per_timestep)}
    evaluator = NativeRegionEvaluator(device, lpips_net="alex", central_fields=fields)
    result = evaluator.evaluate(model, _ut_pipeline(), ds).result

    for key in (
        "central_lpips",
        "full_psnr",
        "full_ssim",
        "central_psnr",
        "central_ssim",
        "peripheral_psnr",
        "peripheral_ssim",
        "lpips",
    ):
        assert key in result.mean, f"missing {key}"
        _assert_finite_metric(result.mean[key])


def _saturating(model, batch, pipeline, sh, device, valid_mask=None):  # noqa: ANN001, ANN202, ARG001
    """A render spanning [-0.2, 1.4] — outside the range any display, or `data_range=1.0`, admits."""
    return batch["images"].to(device) * 1.6 - 0.2


def _preclamped(model, batch, pipeline, sh, device, valid_mask=None):  # noqa: ANN001, ANN202, ARG001
    """The same render, already clamped: what the evaluator should be scoring either way."""
    return (batch["images"].to(device) * 1.6 - 0.2).clamp(0, 1)


@requires_cuda
def test_region_metrics_score_the_clamped_render(
    tmp_fisheye_opencv_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every region metric scores the render clamped to [0, 1]: saturated and pre-clamped renders score identically."""
    device = _device()
    ds = MultiViewSequenceDataset(tmp_fisheye_opencv_dataset, split="train")
    model = FourDGaussians.from_random(n_points=50, sh_degree=0).to(device).eval()

    monkeypatch.setattr(native_evaluator_module, "render_frame_batch", _saturating)
    saturating = NativeRegionEvaluator(device, lpips_net="alex").evaluate(model, None, ds).result
    monkeypatch.setattr(native_evaluator_module, "render_frame_batch", _preclamped)
    preclamped = NativeRegionEvaluator(device, lpips_net="alex").evaluate(model, None, ds).result

    for key, value in saturating.mean.items():
        assert value == pytest.approx(preclamped.mean[key], rel=1e-6), f"{key} differs: the render was not clamped"


def test_clamping_actually_changes_masked_psnr() -> None:
    """Clamping a saturating render raises PSNR, so the clamp test is not vacuous."""
    generator = torch.Generator().manual_seed(5)
    gt = torch.rand(16, 16, 3, generator=generator)
    render = torch.rand(16, 16, 3, generator=generator) * 1.6 - 0.2
    mask = torch.ones(16, 16, dtype=torch.bool)

    raw = masked_psnr(render, gt, mask)
    clamped = masked_psnr(render.clamp(0, 1), gt, mask)
    assert clamped > raw + 0.1


# Observation validity of a native camera's geometry (CPU only).

_CPU = torch.device("cpu")
_FIELD_64 = PinholeField(30.0, 30.0, 31.5, 23.5, 64.0, 48.0)

# Wide KB fisheye: 64x48 frame, fx=fy=20, k1=-0.05. Unlike `tmp_fisheye_opencv_dataset`'s fx=30,
# the corner radius here exceeds the modelled domain, so the corners are genuinely unobserved.
_WIDE_FISHEYE_K = torch.tensor([[20.0, 0.0, 31.5], [0.0, 20.0, 23.5], [0.0, 0.0, 1.0]])
_WIDE_FISHEYE_DIST = torch.tensor([-0.05, 0.0, 0.0, 0.0])


def test_native_geometry_valid_is_model_footprint() -> None:
    """A wide fisheye's `valid` is the camera model's footprint, and the regions live inside it."""
    geom = _native_camera_geometry(
        _WIDE_FISHEYE_K, _WIDE_FISHEYE_DIST, CameraModelType.FISHEYE_OPENCV, 64, 48, _CPU, field=_FIELD_64
    )

    # Footprint and ray-solve validity describe one domain sampled at the pixel centre, so they agree exactly.
    footprint = compute_validity(CameraModelType.FISHEYE_OPENCV, _WIDE_FISHEYE_K, _WIDE_FISHEYE_DIST, 64, 48)
    _, ray_valid = camera_rays(_WIDE_FISHEYE_K, _WIDE_FISHEYE_DIST, CameraModelType.FISHEYE_OPENCV, 64, 48)
    assert torch.equal(geom.valid, footprint)
    assert torch.equal(ray_valid, footprint)

    assert not bool(geom.valid.all())  # the whole point: corners are not observations
    assert not bool(geom.valid[0, 0])
    assert bool(geom.valid[24, 32])  # the centre always is

    # build_region_masks intersects both regions with valid, so neither can score an unobserved
    # pixel, and together they still partition the observed set exactly.
    assert bool((geom.central & ~geom.valid).sum() == 0)
    assert bool((geom.peripheral & ~geom.valid).sum() == 0)
    assert torch.equal(geom.central | geom.peripheral, geom.valid)
    assert bool(geom.peripheral.any())  # the region split stays non-degenerate


def test_native_geometry_central_intersection_is_load_bearing() -> None:
    """A folded ray of an unobserved pixel can read as central, so ``central & valid`` removes pixels."""
    K = torch.tensor([[35.0, 0.0, 31.5], [0.0, 35.0, 23.5], [0.0, 0.0, 1.0]])
    dist = torch.zeros(12)
    dist[0] = -0.3
    field = PinholeField(35.0, 35.0, 31.5, 23.5, 64.0, 48.0)
    geom = _native_camera_geometry(K, dist, CameraModelType.OPENCV, 64, 48, _CPU, field=field)

    rays, _ = camera_rays(K, dist, CameraModelType.OPENCV, 64, 48)
    raw_central = central_mask_from_rays(rays, field)

    assert bool((raw_central & ~geom.valid).any()), "fixture no longer exercises the leak"
    assert bool((geom.central & ~geom.valid).sum() == 0)
    assert bool(geom.central.sum() < raw_central.sum())


def test_native_geometry_pinhole_valid_all_true() -> None:
    """A distortion-free pinhole observes its whole rectangle — unchanged by the validity fix."""
    K = torch.tensor([[50.0, 0.0, 31.5], [0.0, 50.0, 23.5], [0.0, 0.0, 1.0]])
    field = PinholeField(50.0, 50.0, 31.5, 23.5, 64.0, 48.0)
    geom = _native_camera_geometry(K, torch.zeros(12), CameraModelType.PINHOLE, 64, 48, _CPU, field=field)

    assert bool(geom.valid.all())
    assert bool(geom.central.all())  # its own field: everything is central
    assert not bool(geom.peripheral.any())


def test_central_is_the_reference_pinhole_image_not_an_angle_box() -> None:
    """A 150 deg pinhole images up to 75 deg per axis, in tangent; rays at or past 90 deg it never images."""
    field = PinholeField(107.18, 107.18, 400.0, 400.0, 800.0, 800.0)
    angles = torch.tensor([0.0, 74.0, 76.0, 90.0, 100.0]).deg2rad()
    rays = torch.stack([angles.sin(), torch.zeros(5), angles.cos()], dim=-1)[None]
    assert central_mask_from_rays(rays, field).tolist() == [[True, True, False, False, False]]


def test_native_geometry_mild_fisheye_valid_all_true() -> None:
    """A fisheye whose frame fits inside its modelled domain keeps every pixel (no false invalids)."""
    K = torch.tensor([[30.0, 0.0, 31.5], [0.0, 30.0, 23.5], [0.0, 0.0, 1.0]])
    dist = torch.tensor([-0.05, 0.0, 0.0, 0.0])
    geom = _native_camera_geometry(K, dist, CameraModelType.FISHEYE_OPENCV, 64, 48, _CPU, field=_FIELD_64)

    assert bool(geom.valid.all())
    assert bool(geom.peripheral.any())  # the incidence-angle split is untouched by validity


def test_native_geometry_metric_ignores_invalid_corner() -> None:
    """A difference confined to unobserved pixels does not move the score."""
    geom = _native_camera_geometry(
        _WIDE_FISHEYE_K, _WIDE_FISHEYE_DIST, CameraModelType.FISHEYE_OPENCV, 64, 48, _CPU, field=_FIELD_64
    )

    gt = torch.full((48, 64, 3), 0.5)
    render = gt.clone()
    render[~geom.valid] = 0.0  # differ only where nothing was observed

    assert masked_psnr(render, gt, geom.valid) == float("inf")
    assert math.isfinite(masked_psnr(render, gt, torch.ones_like(geom.valid)))  # all-True still scores it


def test_native_geometry_honours_loader_observation_mask() -> None:
    """A reprojected pinhole is scored on the loader's mask, not on its own all-true model footprint."""
    K = torch.tensor([[50.0, 0.0, 31.5], [0.0, 50.0, 23.5], [0.0, 0.0, 1.0]])
    observed = torch.ones(48, 64, dtype=torch.bool)
    observed[:6, :] = False  # the band the source lens never covered
    observed[:, :6] = False

    geom = _native_camera_geometry(K, torch.zeros(12), CameraModelType.PINHOLE, 64, 48, _CPU, observed, field=_FIELD_64)

    assert not bool(geom.valid[:6, :].any())
    assert not bool(geom.central[:, :6].any())
    assert bool(geom.valid[10:, 10:].all())

    gt = torch.full((48, 64, 3), 0.5)
    render = gt.clone()
    render[~observed] = 0.0  # differ only across the unobserved border
    assert masked_psnr(render, gt, geom.valid) == float("inf")


def test_native_geometry_observed_mask_only_narrows() -> None:
    """The loader mask can only remove pixels — it never re-admits any the lens model excluded."""
    args = (_WIDE_FISHEYE_K, _WIDE_FISHEYE_DIST, CameraModelType.FISHEYE_OPENCV, 64, 48, _CPU)
    analytic = _native_camera_geometry(*args)

    all_true = _native_camera_geometry(*args, torch.ones(48, 64, dtype=torch.bool))
    assert bool((all_true.valid == analytic.valid).all())

    partial = torch.ones(48, 64, dtype=torch.bool)
    partial[24:, :] = False
    narrowed = _native_camera_geometry(*args, partial)
    assert bool((narrowed.valid & ~analytic.valid).sum() == 0)
    assert not bool(narrowed.valid[24:, :].any())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
