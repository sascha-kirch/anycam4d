"""Rebuilding the eval model and pipeline from a checkpoint's config snapshot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from plyfile import PlyData
from plyfile import PlyElement
from torch.utils.data import DataLoader

from anycam4d.camera.batched import BatchedCamera
from anycam4d.config.model import ModelConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.enums import AppearanceType
from anycam4d.enums import BackendType
from anycam4d.enums import SamplerType
from anycam4d.eval.checkpoint import build_pipeline_from_checkpoint
from anycam4d.primitives.factory import ModelFactory
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.trainer.trainer import Trainer


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="render path requires CUDA")


def _make_point_cloud_ply(path: Path, n: int = 50) -> None:
    """Write a small random point cloud PLY to *path*."""
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n, 3)).astype("f4")
    arr = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def _render_dict(backend: str = "ewa", post: str = "none") -> dict:
    """A resolved RenderingConfig snapshot (matches RenderingConfig.from_omegaconf keys)."""
    return {
        "backend": backend,
        "near_plane": 0.01,
        "far_plane": 1e10,
        "eps2d": 0.3,
        "packed": False,
        "render_mode": "RGB",
        "rasterize_mode": "classic",
        "post_processor": post,
        "bilagrid_shape": [8, 8, 4],
    }


def _model_dict(
    model_type: str = "fourdgaussians", appearance: str = "sh4d_additive", n_temporal_freqs: int = 4
) -> dict:
    """A resolved ModelConfig snapshot (matches ModelConfig.from_omegaconf keys)."""
    return {
        "model_type": model_type,
        "sh_degree": 0,
        "init_opacity": 0.1,
        "temporal_spread_init": 0.2,
        "prefilter_threshold": 0.05,
        "cov_eps": 1e-6,
        "mu_t_margin_sigmas": -1.0,
        "appearance_model": appearance,
        "n_temporal_freqs": n_temporal_freqs,
    }


def _make_checkpoint(model: FourDGaussians, render: dict, model_cfg: dict | None = None) -> dict:
    """A minimal checkpoint dict around a model + config snapshot (model + rendering)."""
    return {
        "model_state": model.state_dict_with_meta(1.0, torch.zeros(3), 0.0, 1.0),
        "config": {"model": model_cfg or _model_dict(), "rendering": render},
        "norm_scale": 1.0,
        "norm_center": [0.0, 0.0, 0.0],
        "t_min": 0.0,
        "t_max": 1.0,
    }


def _make_trainer(data_dir: Path, result_dir: Path, *, config_snapshot: dict | None) -> Trainer:
    """A tiny EWA trainer over the synthetic dataset (mirrors test_checkpoint)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ply = data_dir / "point_cloud.ply"
    if not ply.exists():
        _make_point_cloud_ply(ply, n=50)

    model_cfg = ModelConfig(sh_degree=0, init_opacity=0.1, temporal_spread_init=0.2, prefilter_threshold=0.0)
    training_cfg = TrainingConfig(
        max_steps=1, sampler=SamplerType.UNIFORM, log_every=999, eval_every=9999, save_every=0
    )
    render_cfg = RenderingConfig()  # EWA + fused CUDA projection (requires CUDA ext)
    strategy = DefaultSpatialStrategy(refine_start_iter=9999, refine_stop_iter=15000, refine_every=100)

    train_ds = MultiViewSequenceDataset(str(data_dir), split="train", scale_factor=1)
    eval_ds = MultiViewSequenceDataset(str(data_dir), split="eval", scale_factor=1)
    norm = compute_normalization(train_ds)
    train_ds.apply_viewmat_transform(norm.T_denorm)
    eval_ds.apply_viewmat_transform(norm.T_denorm)

    model = ModelFactory.create_model(train_ds.point_cloud_path, norm, model_cfg, device)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=anycam4d_collate)
    pipeline = RenderPipeline(backend=BackendFactory.create(render_cfg))

    trainer = Trainer(
        model,
        train_ds,
        eval_ds,
        loader,
        strategy,
        norm,
        training_cfg=training_cfg,
        pipeline=pipeline,
        result_dir=str(result_dir),
        config_snapshot=config_snapshot,
    )
    trainer.setup()
    return trainer


def _render_one(model: FourDGaussians, pipeline: RenderPipeline, data_dir: Path, device: torch.device) -> torch.Tensor:
    """Render a single frame from the first eval sample; returns colors [1, H, W, 3]."""
    ds = MultiViewSequenceDataset(str(data_dir), split="eval", scale_factor=1)
    sample = ds[0]
    H, W = sample.image.shape[:2]
    camera = BatchedCamera.from_single_view(
        sample.viewmat.to(device),
        sample.K.to(device),
        sample.camera_model,
        W,
        H,
        sample.distortion_params.to(device),
    )
    out = pipeline.render(model, camera, sample.timestamp, sh_degree=0)
    return out.colors


@requires_cuda
def test_build_pipeline_ewa_roundtrip(tmp_dataset: Path) -> None:
    """Rebuild model+pipeline from an EWA checkpoint and render one frame."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FourDGaussians.from_random(n_points=200, sh_degree=0, appearance_model=AppearanceType.SH4D_ADDITIVE).to(
        device
    )
    ckpt = _make_checkpoint(model, _render_dict())

    rebuilt, pipeline = build_pipeline_from_checkpoint(ckpt, device)
    assert isinstance(rebuilt, FourDGaussians)
    assert pipeline.post_processor is None

    colors = _render_one(rebuilt, pipeline, tmp_dataset, device)
    assert colors.shape == (1, 48, 64, 3)


def test_build_pipeline_missing_config_raises() -> None:
    """A checkpoint without a config snapshot raises a clear ValueError."""
    model = FourDGaussians.from_random(n_points=10, sh_degree=0, appearance_model=AppearanceType.SH4D_ADDITIVE)
    ckpt = {"model_state": model.state_dict_with_meta(1.0, torch.zeros(3), 0.0, 1.0)}
    with pytest.raises(ValueError, match="no 'config' snapshot"):
        build_pipeline_from_checkpoint(ckpt, torch.device("cpu"))


@requires_cuda
def test_bilagrid_checkpoint_no_post_processor() -> None:
    """A bilateral-grid run rebuilds for eval without a post-processor: the grid only knows training images."""
    device = torch.device("cpu")
    model = FourDGaussians.from_random(n_points=10, sh_degree=0, appearance_model=AppearanceType.SH4D_ADDITIVE)
    ckpt = _make_checkpoint(model, _render_dict(post="bilateral_grid"))

    _, pipeline = build_pipeline_from_checkpoint(ckpt, device)
    assert pipeline.post_processor is None


@pytest.mark.parametrize("cov_eps", [0.0, -1e-6])
def test_model_config_rejects_nonpositive_cov_eps(cov_eps: float) -> None:
    """Kernels divide by the covariance factor's diagonal, which only cov_eps > 0 keeps off zero."""
    with pytest.raises(ValueError, match="cov_eps"):
        ModelConfig(cov_eps=cov_eps)
