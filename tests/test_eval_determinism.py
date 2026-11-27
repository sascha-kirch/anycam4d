"""Eval draws no randomness, so a single ``training.seed`` suffices and runs do not depend on ``eval_every``."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from .test_checkpoint_config import _make_trainer


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Trainer.setup builds a CUDA backend")


def _rng_state() -> tuple:
    """Every global RNG source ``seed_everything`` sets."""
    return (
        torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state_all()[0].clone() if torch.cuda.is_available() else torch.empty(0),
        np.random.get_state()[1].copy(),  # noqa: NPY002 — mirrors what seed_everything seeds
        random.getstate(),
    )


def _states_equal(a: tuple, b: tuple) -> bool:
    return (
        bool(torch.equal(a[0], b[0])) and bool(torch.equal(a[1], b[1])) and np.array_equal(a[2], b[2]) and a[3] == b[3]
    )


@requires_cuda
def test_eval_draws_no_randomness(tmp_dataset: Path, tmp_path: Path) -> None:
    """An eval pass leaves every global RNG stream where it found it."""
    trainer = _make_trainer(tmp_dataset, tmp_path / "t", config_snapshot=None)
    trainer.model.eval()

    before = _rng_state()
    trainer._evaluator.evaluate(trainer.model, trainer.pipeline, trainer.eval_dataset)
    after = _rng_state()

    assert _states_equal(before, after), "eval consumed RNG — it now couples to training and needs its own seed"
    trainer.writer.close()


@requires_cuda
def test_eval_is_reproducible_for_a_fixed_model(tmp_dataset: Path, tmp_path: Path) -> None:
    """Scoring the same model twice gives the same numbers."""
    trainer = _make_trainer(tmp_dataset, tmp_path / "t2", config_snapshot=None)
    trainer.model.eval()

    first = trainer._evaluator.evaluate(trainer.model, trainer.pipeline, trainer.eval_dataset)
    second = trainer._evaluator.evaluate(trainer.model, trainer.pipeline, trainer.eval_dataset)

    # repr is exact for floats and, unlike ==, equates NaN with NaN (a metric undefined on this data).
    assert repr(first.mean) == repr(second.mean)
    assert repr(first.per_timestamp) == repr(second.per_timestamp)
    assert repr(first.per_camera) == repr(second.per_camera)
    trainer.writer.close()
