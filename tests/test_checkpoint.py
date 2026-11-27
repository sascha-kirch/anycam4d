"""Model parameters survive a checkpoint round trip bit for bit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData
from plyfile import PlyElement
from torch.utils.data import DataLoader

from anycam4d.config.model import ModelConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.enums import RasterizeMode
from anycam4d.enums import RenderMode
from anycam4d.enums import SamplerType
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
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    arr = np.zeros(n, dtype=dtype)
    arr["x"] = pts[:, 0]
    arr["y"] = pts[:, 1]
    arr["z"] = pts[:, 2]
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(str(path))


def _make_trainer(data_dir: Path, result_dir: Path) -> Trainer:
    """Create a Trainer with pre-built objects, then call setup()."""
    # The session-scoped tmp_dataset fixture ships no point cloud, so seed one here rather than in
    # each test — otherwise a test only passes when some earlier test happened to create it first.
    ply_path = data_dir / "point_cloud.ply"
    if not ply_path.exists():
        _make_point_cloud_ply(ply_path, n=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig(
        sh_degree=0,
        init_opacity=0.1,
        temporal_spread_init=0.2,
        prefilter_threshold=0.0,
    )
    training_cfg = TrainingConfig(
        max_steps=20,
        sampler=SamplerType.UNIFORM,
        means_lr=1.6e-4,
        quats_lr=1e-3,
        scales_lr=5e-3,
        opacities_lr=5e-2,
        sh0_lr=2.5e-3,
        shN_lr=1.25e-4,
        mu_t_lr=1.6e-5,
        means_lr_final=1.6e-6,
        ssim_start_step=9999,
        log_every=999,
        eval_every=9999,
        save_every=0,
        sh_degree_interval=1000,
        ssim_lambda=0.2,
        mask_bg_alpha_lambda=0.0,
        scale_reg_lambda=0.0,
    )
    render_cfg = RenderingConfig(
        near_plane=0.01,
        far_plane=1e10,
        eps2d=0.3,
        packed=False,
        render_mode=RenderMode.RGB,
        rasterize_mode=RasterizeMode.CLASSIC,
    )
    strategy = DefaultSpatialStrategy(
        refine_start_iter=9999,
        refine_stop_iter=15000,
        refine_every=100,
        reset_every=9999,
    )

    train_ds = MultiViewSequenceDataset(str(data_dir), split="train", scale_factor=1)
    eval_ds = MultiViewSequenceDataset(str(data_dir), split="eval", scale_factor=1)

    norm = compute_normalization(train_ds)
    train_ds.apply_viewmat_transform(norm.T_denorm)
    eval_ds.apply_viewmat_transform(norm.T_denorm)

    model = ModelFactory.create_model(train_ds.point_cloud_path, norm, model_cfg, device)

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=anycam4d_collate)

    pipeline = RenderPipeline(backend=BackendFactory.create(render_cfg))

    trainer = Trainer(
        model,
        train_ds,
        eval_ds,
        train_dataloader,
        strategy,
        norm,
        training_cfg=training_cfg,
        pipeline=pipeline,
        result_dir=str(result_dir),
    )
    trainer.setup()
    return trainer


def test_checkpoint_roundtrip_params(tmp_path: Path) -> None:
    """Save FourDGaussians to a .pt file, reload, verify all parameter tensors match."""
    model = FourDGaussians.from_random(n_points=100, sh_degree=3)

    norm_scale = 2.5
    norm_center = torch.tensor([0.1, -0.2, 0.3])
    t_min = 0.0
    t_max = 1.0

    # Save
    state = model.state_dict_with_meta(norm_scale, norm_center, t_min, t_max)
    ckpt_path = tmp_path / "model.pt"
    torch.save(state, ckpt_path)

    # Load
    loaded_state = torch.load(ckpt_path, weights_only=True)
    model_loaded = FourDGaussians.from_checkpoint(loaded_state)

    # Verify metadata
    assert model_loaded.sh_degree == model.sh_degree
    assert loaded_state["norm_scale"] == norm_scale
    assert loaded_state["norm_center"] == norm_center.tolist()
    assert loaded_state["t_min"] == t_min
    assert loaded_state["t_max"] == t_max

    # Verify every parameter tensor is bit-for-bit identical
    original_params = dict(model.splats)
    loaded_params = dict(model_loaded.splats)
    assert set(original_params.keys()) == set(loaded_params.keys()), (
        f"Parameter key mismatch: {set(original_params.keys())} vs {set(loaded_params.keys())}"
    )
    for name in original_params:
        orig = original_params[name].detach().cpu()
        loaded = loaded_params[name].detach().cpu()
        assert orig.shape == loaded.shape, f"Shape mismatch for '{name}'"
        assert torch.equal(orig, loaded), f"Tensor mismatch for '{name}'"


def test_checkpoint_restores_the_render_constants(tmp_path: Path) -> None:
    """``cov_eps`` and the temporal constants are what the model rendered with, not the constructor defaults."""
    model = FourDGaussians.from_random(n_points=10, sh_degree=0)
    model.cov_eps, model.prefilter_threshold, model.mu_t_margin_sigmas = 3.5e-7, 0.01, 1e6
    torch.save(model.state_dict_with_meta(1.0, torch.zeros(3), 0.0, 1.0), tmp_path / "model.pt")

    loaded = FourDGaussians.from_checkpoint(torch.load(tmp_path / "model.pt", weights_only=True))

    assert (loaded.cov_eps, loaded.prefilter_threshold, loaded.mu_t_margin_sigmas) == (3.5e-7, 0.01, 1e6)
