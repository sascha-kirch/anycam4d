"""Cameras per step (``batch_size``) never moves a learning rate."""

from __future__ import annotations

import pytest
import torch

from anycam4d.config.training import TrainingConfig
from anycam4d.trainer.setup import build_optimizers


class _Splats(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shapes = {
            "means": (4, 3),
            "opacities": (4,),
            "sh0": (4, 1, 3),
            "shN": (4, 15, 3),
            "mu_t": (4,),
            "quats_l": (4, 4),
            "quats_r": (4, 4),
            "scales_4d_log": (4, 4),
        }
        self.splats = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros(s)) for k, s in shapes.items()})


def _lrs(stso: bool, batch_size: int) -> dict[str, float]:
    cfg = TrainingConfig(batch_size=batch_size)
    cfg.selective_adam = stso
    group = build_optimizers(_Splats(), cfg, scene_scale=1.0)
    return {name: opt.param_groups[0]["lr"] for name, opt in group.optimizers.items()}


# The dense and STSO groups are built in separate branches; a scaling applied in one still runs.
@pytest.mark.parametrize("stso", [False, True], ids=["dense", "stso"])
def test_batch_size_does_not_touch_the_learning_rates(stso: bool) -> None:
    assert _lrs(stso, batch_size=1) == _lrs(stso, batch_size=19)
