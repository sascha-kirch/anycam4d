"""Optional eval set and in-loop evaluation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from anycam4d.config.dataset import DatasetConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.data.factory import DatasetFactory
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.enums import SamplerType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.trainer.trainer import Trainer


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="render path requires CUDA")


def _make_trainer(data_dir: Path, result_dir: Path, *, with_eval: bool, eval_every: int) -> Trainer:
    """Minimal set-up Trainer over the fixture dataset with densification effectively off."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_cfg = TrainingConfig(
        max_steps=3,
        sampler=SamplerType.UNIFORM,
        ssim_start_step=9999,
        log_every=999,
        eval_every=eval_every,
        save_every=0,
        sh_degree_interval=1000,
        ssim_lambda=0.2,
        mask_bg_alpha_lambda=0.0,
        scale_reg_lambda=0.0,
    )
    render_cfg = RenderingConfig()
    strategy = DefaultSpatialStrategy(
        refine_start_iter=9999, refine_stop_iter=15000, refine_every=100, reset_every=9999
    )

    train_ds = MultiViewSequenceDataset(str(data_dir), split="train", scale_factor=1)
    eval_ds = MultiViewSequenceDataset(str(data_dir), split="eval", scale_factor=1) if with_eval else None

    norm = compute_normalization(train_ds)
    train_ds.apply_viewmat_transform(norm.T_denorm)
    if eval_ds is not None:
        eval_ds.apply_viewmat_transform(norm.T_denorm)

    model = FourDGaussians.from_random(n_points=500, sh_degree=0).to(device)
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


def test_create_datasets_eval_optional(tmp_dataset: Path, tmp_path: Path) -> None:
    """A data_dir with only train_dataset.json yields (train, None)."""
    train_only = tmp_path / "train_only"
    shutil.copytree(tmp_dataset, train_only)
    (train_only / "eval_dataset.json").unlink()

    train, eval_ds = DatasetFactory.create_datasets(DatasetConfig(data_dir=train_only))
    assert isinstance(train, MultiViewSequenceDataset)
    assert eval_ds is None


def test_create_datasets_eval_present(tmp_dataset: Path) -> None:
    """A data_dir with both JSONs yields two non-None datasets."""
    train, eval_ds = DatasetFactory.create_datasets(DatasetConfig(data_dir=tmp_dataset))
    assert isinstance(train, MultiViewSequenceDataset)
    assert isinstance(eval_ds, MultiViewSequenceDataset)


@requires_cuda
def test_trainer_runs_without_eval_set(tmp_dataset: Path, tmp_path: Path) -> None:
    """Trainer(eval_dataset=None) completes; no evaluator, no eval/* scalars."""
    trainer = _make_trainer(tmp_dataset, tmp_path / "run", with_eval=False, eval_every=1)
    logged: list[str] = []
    trainer.writer.add_scalar = lambda tag, *a, **k: logged.append(tag)  # type: ignore[method-assign]

    trainer.train()

    assert trainer._evaluator is None
    assert not any(tag.startswith("eval/") for tag in logged)


@requires_cuda
def test_trainer_runs_eval(tmp_dataset: Path, tmp_path: Path) -> None:
    """Eval set present + eval_every=1 logs eval/psnr|ssim|lpips (+ _std)."""
    trainer = _make_trainer(tmp_dataset, tmp_path / "run", with_eval=True, eval_every=1)
    logged: list[str] = []
    trainer.writer.add_scalar = lambda tag, *a, **k: logged.append(tag)  # type: ignore[method-assign]

    trainer.train()

    assert trainer._evaluator is not None
    for metric in ("eval/psnr", "eval/ssim", "eval/lpips"):
        assert metric in logged
        assert f"{metric}_std" in logged


@requires_cuda
def test_eval_every_zero_disables(tmp_dataset: Path, tmp_path: Path) -> None:
    """eval_every=0 disables in-loop eval even when an eval set is present."""
    trainer = _make_trainer(tmp_dataset, tmp_path / "run", with_eval=True, eval_every=0)
    calls: list[int] = []
    trainer._run_eval = lambda step: calls.append(step)  # type: ignore[method-assign]

    trainer.train()

    assert calls == []


@requires_cuda
def test_eval_restores_train_mode(tmp_dataset: Path, tmp_path: Path) -> None:
    """model.training is True after a _run_eval call."""
    trainer = _make_trainer(tmp_dataset, tmp_path / "run", with_eval=True, eval_every=999)
    trainer.model.train()
    trainer._run_eval(0)
    assert trainer.model.training is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
