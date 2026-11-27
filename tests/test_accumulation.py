"""Gradient accumulation: N renders per optimizer step, and the quantities that must not follow N."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.data.factory import DatasetFactory
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization
from anycam4d.data.samplers import FrameAlignedSampler
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.enums import SamplerType
from anycam4d.enums import SpatialTrigger
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.strategy.coupled import OGFourDGSStrategy
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.strategy.mcmc import MCMCSpatialStrategy
from anycam4d.strategy.null_strategy import NullStrategy
from anycam4d.trainer.trainer import MicroBatch
from anycam4d.trainer.trainer import Trainer


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="render path requires CUDA")


def _build_trainer(
    tmp_dataset: Path,
    tmp_path: Path,
    strategy: object,
    *,
    accum_renders: int = 1,
    split_in_time: bool = False,
    opacity_reg_lambda: float = 0.0,
) -> Trainer:
    """Frame-aligned trainer on the synthetic dataset (2 frames x 3 cameras), loader order fixed.

    ``split_in_time`` gates each half of the primitives onto one of the two frames, so the two renders
    of a step update disjoint sets.
    """
    device = torch.device("cuda")

    train_ds = MultiViewSequenceDataset(tmp_dataset, split="train")
    norm = compute_normalization(train_ds)
    train_ds.apply_viewmat_transform(norm.T_denorm)

    model = FourDGaussians.from_random(n_points=300, sh_degree=0).to(device)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    with torch.no_grad():
        # Identity rotations make the temporal marginal exp(2 * scales_4d_log[:, 3]), so the gate below
        # is set by that one parameter rather than by the random 4D coupling.
        model.splats["quats_l"][:] = identity
        model.splats["quats_r"][:] = identity
        if split_in_time:
            half = len(model) // 2
            model.splats["mu_t"][:half] = -0.5
            model.splats["mu_t"][half:] = 0.5
            model.splats["scales_4d_log"][:, 3] = math.log(0.1)
        else:
            model.splats["mu_t"].zero_()
            model.splats["scales_4d_log"][:, 3] = 0.5

    loader = DataLoader(
        train_ds,
        batch_sampler=FrameAlignedSampler(train_ds, shuffle=False),
        collate_fn=anycam4d_collate,
    )
    training_cfg = TrainingConfig(
        max_steps=4,
        sampler=SamplerType.FRAME_ALIGNED,
        accum_renders=accum_renders,
        ssim_start_step=9999,
        log_every=9999,
        eval_every=0,
        save_every=0,
        save_best=False,
        mask_bg_alpha_lambda=0.0,
        scale_reg_lambda=0.0,
        opacity_reg_lambda=opacity_reg_lambda,
    )

    trainer = Trainer(
        model,
        train_ds,
        None,
        loader,
        strategy,
        norm,
        training_cfg=training_cfg,
        pipeline=RenderPipeline(backend=BackendFactory.create(RenderingConfig())),
        result_dir=str(tmp_path / "result"),
    )
    trainer.setup()
    return trainer


def _capture_step(trainer: Trainer, step: int) -> tuple[Tensor, Tensor]:
    """Run one step with the optimizer stubbed out, so parameters stay fixed across phases.

    Returns the accumulated ``means`` gradient and the activity mask the step would have stepped.
    """
    captured: dict[str, Tensor] = {}

    def fake_optimizer_step(active: Tensor) -> None:
        captured["active"] = active.clone()
        captured["means_grad"] = trainer.model.splats["means"].grad.detach().clone()

    trainer._optimizer_step = fake_optimizer_step  # type: ignore[method-assign]
    trainer.schedulers = []  # nothing steps the optimizers, so nothing may step their schedulers
    trainer._train_step_body(step)
    return captured["means_grad"], captured["active"]


def _capture_scalars(trainer: Trainer, step: int) -> dict[str, float]:
    """Run one step with the optimizer stubbed out and return the per-render scalars it logged."""
    logged: dict[str, float] = {}
    writer = trainer.writer

    def collect(tag: str, value: float, global_step: int | None = None) -> None:  # noqa: ARG001
        if tag == "train/psnr":
            logged[tag] = float(value)

    trainer.writer = type(
        "W",
        (),
        {
            "add_scalar": staticmethod(collect),
            "add_histogram": staticmethod(lambda *a, **k: None),
            "add_image": staticmethod(lambda *a, **k: None),
        },
    )()
    try:
        _capture_step(trainer, step)
    finally:
        trainer.writer = writer
    return logged


def _rewind(trainer: Trainer, accum_renders: int) -> None:
    """Replay the same frames at a different accumulation depth."""
    trainer.training_cfg.accum_renders = accum_renders
    trainer._trainloader_iter = iter(trainer._trainloader)


@requires_cuda
def test_accumulated_grad_is_the_mean_of_its_renders(tmp_dataset: Path, tmp_path: Path) -> None:
    """One step over N renders carries the mean of the N per-render gradients, not their sum."""
    trainer = _build_trainer(tmp_dataset, tmp_path, NullStrategy(), accum_renders=2)
    accumulated, _ = _capture_step(trainer, 0)

    _rewind(trainer, 1)
    first, _ = _capture_step(trainer, 0)
    second, _ = _capture_step(trainer, 1)

    reference = (first + second) / 2.0
    rel_err = (accumulated - reference).abs().max() / reference.abs().max()
    assert rel_err < 1e-5, f"accumulated gradient is not the mean of its renders (rel err {rel_err:.2e})"


@requires_cuda
@pytest.mark.parametrize("n_renders", [1, 3])
def test_step_draws_exactly_accum_renders_batches(tmp_dataset: Path, tmp_path: Path, n_renders: int) -> None:
    """A step consumes N loader batches — at N=1 the data consumption of every shipped run is unchanged."""
    trainer = _build_trainer(tmp_dataset, tmp_path, NullStrategy(), accum_renders=n_renders)
    draws = []
    pull = trainer._next_micro_batch

    def counting_pull():
        draws.append(1)
        return pull()

    trainer._next_micro_batch = counting_pull  # type: ignore[method-assign]
    _capture_step(trainer, 0)

    assert len(draws) == n_renders


@requires_cuda
@pytest.mark.parametrize(
    ("cls", "trigger", "grad_key", "count_key"),
    [
        (DefaultSpatialStrategy, SpatialTrigger.PARAMSPACE, "grad3d", "count3d"),
        (DefaultSpatialStrategy, SpatialTrigger.SCREENSPACE, "grad2d", "count"),
    ],
)
def test_densification_statistic_does_not_follow_accum_renders(
    tmp_dataset: Path, tmp_path: Path, cls: type, trigger: SpatialTrigger, grad_key: str, count_key: str
) -> None:
    """N renders inside one step accumulate what the same N renders accumulate across N steps.

    Without the ``grad_scale`` rescale the signal would be N times smaller, silently making every growth
    threshold N times stricter; without the per-render ``means_grad`` the first render would be
    counted N times.
    """
    strategy = cls(
        spatial_trigger=trigger,
        refine_start_iter=10_000,  # no refine inside the window this test uses
        refine_stop_iter=15_000,
        refine_every=1000,
        reset_every=99_999,
    )
    trainer = _build_trainer(tmp_dataset, tmp_path, strategy, accum_renders=2)
    _capture_step(trainer, 0)

    accumulated = trainer.strategy_state[grad_key].clone()
    counts = trainer.strategy_state[count_key].clone()

    trainer.strategy_state = strategy.initialize_state(scene_scale=trainer.normalization.scene_scale)
    _rewind(trainer, 1)
    _capture_step(trainer, 0)
    _capture_step(trainer, 1)

    assert accumulated.abs().max() > 0, "the fixture accumulated no signal, so the comparison is vacuous"
    torch.testing.assert_close(trainer.strategy_state[grad_key], accumulated, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(trainer.strategy_state[count_key], counts)


@requires_cuda
def test_activity_mask_is_the_union_over_the_steps_renders(tmp_dataset: Path, tmp_path: Path) -> None:
    """The optimizer steps every primitive any render updated; masking by the last render drops the rest."""
    trainer = _build_trainer(tmp_dataset, tmp_path, NullStrategy(), accum_renders=2, split_in_time=True)
    _, union = _capture_step(trainer, 0)

    _rewind(trainer, 1)
    _, first = _capture_step(trainer, 0)
    _, second = _capture_step(trainer, 1)

    assert first.sum() < union.sum(), "the temporal gate did not separate the two renders; test is vacuous"
    assert second.sum() < union.sum()
    assert torch.equal(union, first | second)


@requires_cuda
@pytest.mark.parametrize("family", ["mcmc_spatial", "default_spatial", "og_4dgs"])
def test_accumulated_step_runs_for_every_densifier_family(tmp_dataset: Path, tmp_path: Path, family: str) -> None:
    """Two real steps at accum_renders=4, optimizer and densifier live, stay finite."""
    reg = 0.0
    if family == "mcmc_spatial":
        strategy = MCMCSpatialStrategy(refine_start_iter=0, refine_every=1, cap_max=400)
        reg = 0.002
    elif family == "og_4dgs":
        strategy = OGFourDGSStrategy(
            spatial_trigger=SpatialTrigger.SCREENSPACE,
            refine_start_iter=0,
            refine_every=1,
            reset_every=99_999,
        )
    else:
        strategy = DefaultSpatialStrategy(refine_start_iter=0, refine_every=1, reset_every=99_999)

    trainer = _build_trainer(tmp_dataset, tmp_path, strategy, accum_renders=4, opacity_reg_lambda=reg)
    trainer._train_step_body(1)
    trainer._train_step_body(2)

    for name, param in trainer.model.splats.items():
        assert param.isfinite().all(), f"{name} went non-finite under accumulation"


@requires_cuda
def test_step_rejects_renders_of_different_resolutions(tmp_dataset: Path, tmp_path: Path) -> None:
    """A step whose renders disagree on resolution raises rather than averaging two pixel scales."""
    trainer = _build_trainer(tmp_dataset, tmp_path, NullStrategy(), accum_renders=2)
    pull = trainer._next_micro_batch
    seen: list[int] = []

    def shrinking_pull() -> MicroBatch:
        micro = pull()
        seen.append(1)
        if len(seen) == 1:
            return micro
        half = micro.images[:, : micro.images.shape[1] // 2, : micro.images.shape[2] // 2]
        return replace(micro, images=half, masks=None, validity=None)

    trainer._next_micro_batch = shrinking_pull  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="must share one resolution"):
        trainer._train_step_body(0)


@requires_cuda
def test_logged_scalars_are_the_steps_mean_not_the_last_render(tmp_dataset: Path, tmp_path: Path) -> None:
    """Every series a decision is read on averages the step's renders; the last render alone is a quarter of it."""
    trainer = _build_trainer(tmp_dataset, tmp_path, NullStrategy(), accum_renders=2)
    accumulated = _capture_scalars(trainer, 0)

    # Both replays run at step 0: the logging gate is `step % log_every`, and this fixture logs only there.
    _rewind(trainer, 1)
    first = _capture_scalars(trainer, 0)
    second = _capture_scalars(trainer, 0)

    assert abs(first["train/psnr"] - second["train/psnr"]) > 1e-3, "the two renders scored alike; test is vacuous"
    for name, value in accumulated.items():
        assert value == pytest.approx((first[name] + second[name]) / 2.0, rel=1e-5), f"{name} is not the step mean"
    assert "train/psnr" in accumulated


@requires_cuda
def test_temporal_tiled_steps_stay_on_tile_boundaries_across_epochs(tmp_dataset: Path, tmp_path: Path) -> None:
    """Every step renders one camera's consecutive frames, through the real loader and its epoch restart.

    A drift by one render would pair two cameras in a step and quietly turn the sampler back into a shuffle.
    """
    n_renders = 2
    trainer = _build_trainer(tmp_dataset, tmp_path, NullStrategy(), accum_renders=n_renders)
    trainer.training_cfg.sampler = SamplerType.TEMPORAL_TILED
    trainer._trainloader = DatasetFactory.create_dataloader(trainer.train_dataset, trainer.training_cfg)
    trainer._trainloader_iter = iter(trainer._trainloader)
    ds = trainer.train_dataset

    steps_per_epoch = len(ds) // n_renders
    for _ in range(2 * steps_per_epoch + 1):
        micros = [trainer._next_micro_batch() for _ in range(n_renders)]
        cams = {m.camera_ids[0] for m in micros}
        frames = [ds.frame_ids.index(ds.frame_id_for(m.image_ids[0])) for m in micros]
        assert len(cams) == 1
        assert frames == list(range(frames[0], frames[0] + n_renders))
