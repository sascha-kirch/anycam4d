"""Frame-aligned batches: collate, C>1 render equivalence, the training step."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from anycam4d.camera.batched import BatchedCamera
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization
from anycam4d.data.samplers import FrameAlignedSampler
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.data.types import TrainingSample
from anycam4d.enums import CameraModelType
from anycam4d.enums import SamplerType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.trainer.trainer import Trainer


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="render path requires CUDA")

_H, _W = 32, 32


def _sample(camera_model: CameraModelType, dist_width: int, *, with_mask: bool = True) -> TrainingSample:
    """One synthetic TrainingSample on CPU."""
    return TrainingSample(
        image=torch.rand(_H, _W, 3),
        K=torch.eye(3),
        viewmat=torch.eye(4),
        timestamp=0.0,
        image_id=0,
        camera_id=0,
        camera_model=camera_model,
        distortion_params=torch.zeros(dist_width),
        mask=torch.ones(_H, _W, 1) if with_mask else None,
    )


def test_collate_stacks_leading_C() -> None:
    """anycam4d_collate stacks every per-view field with a leading dim C."""
    c = 4
    samples = [_sample(CameraModelType.PINHOLE, 12) for _ in range(c)]
    batch = anycam4d_collate(samples)

    assert batch["viewmats"].shape == (c, 4, 4)
    assert batch["Ks"].shape == (c, 3, 3)
    assert batch["distortion_params"].shape == (c, 12)
    assert batch["images"].shape == (c, _H, _W, 3)
    assert batch["masks"].shape == (c, _H, _W, 1)
    assert len(batch["timestamps"]) == c
    assert batch["camera_model"] == CameraModelType.PINHOLE


def test_collate_masks_none_when_any_missing() -> None:
    """anycam4d_collate -> masks=None if any sample lacks a mask."""
    samples = [_sample(CameraModelType.PINHOLE, 12), _sample(CameraModelType.PINHOLE, 12, with_mask=False)]
    assert anycam4d_collate(samples)["masks"] is None


def test_collate_rejects_mixed_models() -> None:
    """A batch mixing camera models raises ValueError."""
    samples = [_sample(CameraModelType.PINHOLE, 12), _sample(CameraModelType.FISHEYE_OPENCV, 4)]
    with pytest.raises(ValueError, match="mixed camera models"):
        anycam4d_collate(samples)


def test_collate_rejects_mixed_resolutions() -> None:
    """A batch mixing image resolutions raises ValueError naming the shapes."""
    s0 = _sample(CameraModelType.PINHOLE, 12)
    s1 = TrainingSample(
        image=torch.rand(_H + 8, _W, 3),  # different H
        K=torch.eye(3),
        viewmat=torch.eye(4),
        timestamp=0.0,
        image_id=1,
        camera_id=0,
        camera_model=CameraModelType.PINHOLE,
        distortion_params=torch.zeros(12),
        mask=torch.ones(_H + 8, _W, 1),
    )
    with pytest.raises(ValueError, match="mixed image resolutions"):
        anycam4d_collate([s0, s1])


def _batched_camera(c: int, device: torch.device) -> BatchedCamera:
    """A C-camera pinhole carrier with distinct extrinsics."""
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(c, 1, 1)
    for i in range(c):
        viewmats[i, 2, 3] = 3.0 + 0.5 * i  # vary the camera distance per view
    cx, cy = _W / 2.0, _H / 2.0
    k = torch.tensor([[50.0, 0.0, cx], [0.0, 50.0, cy], [0.0, 0.0, 1.0]], device=device)
    ks = k.unsqueeze(0).repeat(c, 1, 1)
    dist = torch.zeros(c, 12, device=device)
    return BatchedCamera(viewmats=viewmats, Ks=ks, model=CameraModelType.PINHOLE, width=_W, height=_H, distortion=dist)


@requires_cuda
def test_C_gt_1_equals_C1_loop(random_model: FourDGaussians) -> None:
    """One C-camera render equals C separate C=1 renders (batching is not a behaviour change)."""
    device = next(random_model.parameters()).device
    c = 3
    cam = _batched_camera(c, device)
    pipeline = RenderPipeline(backend=BackendFactory.create(RenderingConfig()))

    with torch.no_grad():
        out_batched = pipeline.render(random_model, cam, timestamp=0.0, sh_degree=random_model.sh_degree)

        for i in range(c):
            single = BatchedCamera(
                viewmats=cam.viewmats[i : i + 1],
                Ks=cam.Ks[i : i + 1],
                model=cam.model,
                width=cam.width,
                height=cam.height,
                distortion=cam.distortion[i : i + 1],
            )
            out_single = pipeline.render(random_model, single, timestamp=0.0, sh_degree=random_model.sh_degree)
            assert torch.allclose(out_batched.colors[i], out_single.colors[0], atol=1e-5), f"camera {i} colors differ"
            assert torch.allclose(out_batched.alphas[i], out_single.alphas[0], atol=1e-5), f"camera {i} alphas differ"


def _training_cfg() -> TrainingConfig:
    return TrainingConfig(
        max_steps=2,
        sampler=SamplerType.FRAME_ALIGNED,
        ssim_start_step=9999,
        log_every=999,
        eval_every=9999,
        save_every=0,
        mask_bg_alpha_lambda=0.2,
        scale_reg_lambda=0.0,
    )


def _build_trainer(tmp_dataset: Path, tmp_path: Path, collate: Callable[[list[TrainingSample]], dict]) -> Trainer:
    """Frame-aligned C=3 trainer on the synthetic dataset, ready for one _train_step_body."""
    device = torch.device("cuda")

    train_ds = MultiViewSequenceDataset(tmp_dataset, split="train")
    eval_ds = MultiViewSequenceDataset(tmp_dataset, split="eval")
    norm = compute_normalization(train_ds)
    train_ds.apply_viewmat_transform(norm.T_denorm)
    eval_ds.apply_viewmat_transform(norm.T_denorm)

    assert train_ds.cameras_per_timestep == 3  # the C>1 invariant for this fixture

    model = FourDGaussians.from_random(n_points=500, sh_degree=0).to(device)
    # Widen the temporal scale + centre mu_t so the temporal gate does not zero out the render.
    with torch.no_grad():
        model.splats["scales_4d_log"][:, 3] = 0.5
        model.splats["mu_t"].zero_()

    sampler = FrameAlignedSampler(train_ds, shuffle=False)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate)

    strategy = DefaultSpatialStrategy(refine_start_iter=0, refine_stop_iter=15000, refine_every=1, reset_every=9999)
    pipeline = RenderPipeline(backend=BackendFactory.create(RenderingConfig()))

    trainer = Trainer(
        model,
        train_ds,
        eval_ds,
        loader,
        strategy,
        norm,
        training_cfg=_training_cfg(),
        pipeline=pipeline,
        result_dir=str(tmp_path / "result"),
    )
    trainer.setup()
    return trainer


@requires_cuda
def test_train_step_C_gt_1_smoke(tmp_dataset: Path, tmp_path: Path) -> None:
    """One _train_step_body on a frame-aligned C=3 batch: finite loss, grad populated, strategy runs."""
    trainer = _build_trainer(tmp_dataset, tmp_path, anycam4d_collate)
    model = trainer.model

    # _train_step_body zeroes grads (set_to_none) at its end, so capture means.grad mid-backward.
    captured: dict[str, torch.Tensor] = {}
    handle = model.splats["means"].register_hook(lambda g: captured.__setitem__("means_grad", g.detach().clone()))
    trainer._train_step_body(1)  # step>refine_start_iter so the strategy densify path also runs
    handle.remove()

    assert "means_grad" in captured, "backward did not populate means.grad"
    assert captured["means_grad"].isfinite().all()
    # Strategy step ran without a C-shape error and parameters stayed finite.
    assert model.splats["means"].isfinite().all()


def _collate_with_validity(samples: list[TrainingSample]) -> dict:
    """anycam4d_collate, then inject a half-invalid observation mask (the fixture is all-valid)."""
    batch = anycam4d_collate(samples)
    c, h, w, _ = batch["images"].shape
    validity = torch.ones(c, h, w, 1, dtype=torch.bool)
    validity[:, h // 2 :] = False
    batch["validity"] = validity
    return batch


@requires_cuda
def test_train_step_with_validity(tmp_dataset: Path, tmp_path: Path) -> None:
    """A non-trivial batch["validity"] drives the masked loss path through a full training step."""
    trainer = _build_trainer(tmp_dataset, tmp_path, _collate_with_validity)
    model = trainer.model

    captured: dict[str, torch.Tensor] = {}
    handle = model.splats["means"].register_hook(lambda g: captured.__setitem__("means_grad", g.detach().clone()))
    trainer._train_step_body(1)
    handle.remove()

    assert "means_grad" in captured, "backward did not populate means.grad on the masked path"
    assert captured["means_grad"].isfinite().all()
    assert model.splats["means"].isfinite().all()
