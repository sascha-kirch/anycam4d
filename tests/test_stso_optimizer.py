"""The spatio-temporal sparse optimizer (Selective Adam), its group manager, selection and a training run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gsplat.optimizers import SelectiveAdam
from jaxtyping import Float
from plyfile import PlyData
from plyfile import PlyElement
from torch import Tensor
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
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.trainer.optimizers.group import DenseAdamGroup
from anycam4d.trainer.optimizers.group import SpatioTemporalSparseAdamGroup
from anycam4d.trainer.optimizers.stso import SpatioTemporalSparseAdamFused
from anycam4d.trainer.optimizers.stso import SpatioTemporalSparseAdamReference
from anycam4d.trainer.setup import build_optimizers
from anycam4d.trainer.trainer import Trainer


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="render path requires CUDA")


BETAS = (0.9, 0.999)
EPS = 1e-15
LR = 1e-2


def _ref_opt(p: Float[Tensor, "n ..."]) -> SpatioTemporalSparseAdamReference:
    return SpatioTemporalSparseAdamReference([{"params": p, "lr": LR}], lr=LR, betas=BETAS, eps=EPS)


def _fused_opt(p: Float[Tensor, "n ..."]) -> SpatioTemporalSparseAdamFused:
    return SpatioTemporalSparseAdamFused([{"params": p, "lr": LR}], lr=LR, betas=BETAS, eps=EPS)


_SHAPES = [(7, 3), (7, 1), (7,), (7, 4), (7, 5, 3)]


@requires_cuda
@pytest.mark.parametrize("make_opt", [_ref_opt, _fused_opt], ids=["reference", "fused"])
@pytest.mark.parametrize("shape", _SHAPES, ids=[str(s) for s in _SHAPES])
def test_matches_gsplat_selective_adam(shape: tuple[int, ...], make_opt) -> None:  # noqa: ANN001
    """Both variants match gsplat's published ``SelectiveAdam`` over every parameter shape, several steps
    and a partial mask: parameters and both moments."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    p0 = torch.randn(*shape, device=device)
    p_ours = torch.nn.Parameter(p0.clone())
    p_gs = torch.nn.Parameter(p0.clone())
    ours = make_opt(p_ours)
    gs = SelectiveAdam([{"params": p_gs, "lr": LR}], eps=EPS, betas=BETAS)

    n = shape[0]
    for _ in range(5):
        vis = torch.rand(n, device=device) > 0.3
        g = torch.randn(*shape, device=device)
        p_ours.grad = g.clone()
        p_gs.grad = g.clone()
        ours.step(vis)
        gs.step(vis)

    assert torch.allclose(p_ours, p_gs, atol=1e-5)
    so, sg = ours.state[p_ours], gs.state[p_gs]
    assert torch.allclose(so["exp_avg"], sg["exp_avg"], atol=1e-5)
    assert torch.allclose(so["exp_avg_sq"], sg["exp_avg_sq"], atol=1e-5)


def test_inactive_primitive_is_frozen(random_model, make_grads) -> None:
    """An inactive primitive keeps its parameters and both moments unchanged for the step."""
    model = random_model
    n = len(model)
    device = model.splats["means"].device
    group = SpatioTemporalSparseAdamGroup({"means": _ref_opt(model.splats["means"])})

    make_grads(model)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    mask[0] = False

    theta_before = model.splats["means"].detach()[0].clone()
    group.step(visibility=mask)

    state = group.optimizers["means"].state[model.splats["means"]]
    assert torch.equal(model.splats["means"].detach()[0], theta_before)
    assert torch.count_nonzero(state["exp_avg"][0]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"][0]) == 0
    assert torch.count_nonzero(state["exp_avg"][1]) > 0


def _training_cfg(*, selective_adam: bool = False) -> TrainingConfig:
    return TrainingConfig(max_steps=10, selective_adam=selective_adam)


def test_disabled_builds_dense_group_matching_adam(random_model, make_grads) -> None:
    """When the sparse optimizer is disabled, ``build_optimizers`` returns a dense group whose step
    matches a plain ``torch.optim.Adam`` step on the same gradients."""
    model = random_model
    group = build_optimizers(model, _training_cfg(selective_adam=False))
    assert isinstance(group, DenseAdamGroup)

    means = model.splats["means"]
    p_ref = torch.nn.Parameter(means.detach().clone())
    lr = group.optimizers["means"].param_groups[0]["lr"]
    eps = group.optimizers["means"].param_groups[0]["eps"]
    adam = torch.optim.Adam([p_ref], lr=lr, betas=BETAS, eps=eps)

    make_grads(model)
    p_ref.grad = means.grad.clone()
    group.step(visibility=None)
    adam.step()
    assert torch.allclose(means.detach(), p_ref.detach(), atol=1e-6)


@pytest.mark.parametrize("has_ext", [True, False], ids=["ext", "no_ext"])
def test_enabled_builds_fused_or_falls_back(random_model, monkeypatch, has_ext: bool) -> None:
    """Enabled builds the fused optimizer; without the CUDA extension it warns and uses the reference."""
    if has_ext and not torch.cuda.is_available():
        pytest.skip("fused kernel requires CUDA")
    monkeypatch.setattr("anycam4d.trainer.setup.HAS_CUDA_EXT", has_ext)
    expected = SpatioTemporalSparseAdamFused if has_ext else SpatioTemporalSparseAdamReference
    if has_ext:
        group = build_optimizers(random_model, _training_cfg(selective_adam=True))
    else:
        with pytest.warns(UserWarning, match="fused STSO optimizer requires the CUDA extension"):
            group = build_optimizers(random_model, _training_cfg(selective_adam=True))
    assert isinstance(group, SpatioTemporalSparseAdamGroup)
    assert all(isinstance(opt, expected) for opt in group.optimizers.values())


def _build_visible_dataset(base: Path) -> None:
    """2-frame, 3-camera dataset whose points sit in front of the cameras, so the active set is non-empty."""
    from PIL import Image  # noqa: PLC0415 — test-only, keeps the import off the module hot path

    (base / "images").mkdir(parents=True)
    rng = np.random.default_rng(1)
    cams = [
        {"camera_id": c, "model": "pinhole", "width": 64, "height": 48, "params": [50.0, 50.0, 32.0, 24.0]}
        for c in range(3)
    ]
    caps = []
    for fi, t in enumerate([0.0, 1.0 / 30.0]):
        for c in range(3):
            rgb = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            a = rng.choice(np.array([0, 255], dtype=np.uint8), size=(48, 64, 1))
            Image.fromarray(np.concatenate([rgb, a], 2), "RGBA").save(base / "images" / f"cam{c:02d}_{fi:04d}.png")
            caps.append(
                {
                    "image_path": f"images/cam{c:02d}_{fi:04d}.png",
                    "camera_id": c,
                    "frame_id": fi,
                    "timestamp": t,
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [c * 0.5, 0.0, 3.0],
                }
            )
    meta = {
        "world_coordinate_system": "right_handed",
        "camera_coordinate_system": "opencv",
        "transform_convention": "camera_to_world",
    }
    (base / "train_dataset.json").write_text(
        json.dumps({"schema_version": 1, "metadata": meta, "cameras": cams, "captures": caps})
    )

    pts = (rng.standard_normal((50, 3)) * 0.3 + np.array([0.0, 0.0, 6.0])).astype("f4")
    arr = np.zeros(50, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(base / "point_cloud.ply"))


@requires_cuda
def test_sparse_optimizer_trains_through_densification(tmp_path: Path) -> None:
    """A sparse-optimizer run densifies every refine step and its Adam moments stay row-aligned with the primitives."""
    data_dir = tmp_path / "seq"
    _build_visible_dataset(data_dir)

    device = torch.device("cuda")
    model_cfg = ModelConfig(sh_degree=0, init_opacity=0.1, temporal_spread_init=0.2, prefilter_threshold=0.05)
    training_cfg = TrainingConfig(
        max_steps=40,
        sampler=SamplerType.UNIFORM,
        ssim_start_step=9999,
        log_every=999,
        eval_every=9999,
        save_every=0,
        mask_bg_alpha_lambda=0.0,
        selective_adam=True,
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
        refine_start_iter=10,
        refine_stop_iter=35,
        refine_every=10,
        reset_every=9999,
        grow_grad2d=0.0,
        grow_grad3d=0.0,
        prune_opa=0.0,
    )
    train_ds = MultiViewSequenceDataset(str(data_dir), split="train", scale_factor=1)
    norm = compute_normalization(train_ds)
    train_ds.apply_viewmat_transform(norm.T_denorm)
    model = ModelFactory.create_model(train_ds.point_cloud_path, norm, model_cfg, device)
    dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=anycam4d_collate)
    pipeline = RenderPipeline(backend=BackendFactory.create(render_cfg))

    trainer = Trainer(
        model,
        train_ds,
        None,
        dl,
        strategy,
        norm,
        training_cfg=training_cfg,
        pipeline=pipeline,
        result_dir=str(tmp_path / "stso"),
    )
    trainer.setup()
    assert isinstance(trainer.optimizers, SpatioTemporalSparseAdamGroup)

    n0 = len(model)
    for step in range(40):
        trainer._train_step(step)

    assert len(model) > n0, "expected a grow once the growth-gradient thresholds are zero"
    for opt in trainer.optimizers.optimizers.values():
        state = opt.state[opt.param_groups[0]["params"][0]]
        assert state["exp_avg"].shape[0] == state["exp_avg_sq"].shape[0] == len(model)
