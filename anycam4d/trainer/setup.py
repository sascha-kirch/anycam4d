"""Optimizer and scheduler setup for Trainer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.trainer.optimizers.group import DenseAdamGroup
from anycam4d.trainer.optimizers.group import SpatioTemporalSparseAdamGroup
from anycam4d.trainer.optimizers.stso import SpatioTemporalSparseAdamFused
from anycam4d.trainer.optimizers.stso import SpatioTemporalSparseAdamReference


if TYPE_CHECKING:
    from anycam4d.config.training import TrainingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.bilateral_grid_post_processor import BilateralGridPostProcessor
    from anycam4d.trainer.optimizers.group import OptimizerGroupProto
    from anycam4d.trainer.optimizers.stso import SpatioTemporalSparseAdam


def build_optimizers(
    model: FourDGaussians,
    training_cfg: TrainingConfig,
    *,
    scene_scale: float = 1.0,
) -> OptimizerGroupProto:
    """One optimizer per ``model.splats`` parameter, keyed by name so densification ops resize them.

    ``training.selective_adam`` selects the sparse group, else dense Adam.
    """
    t = training_cfg
    betas = tuple(t.optimizer_betas)
    lr_map: dict[str, float] = {
        "means": t.means_lr * scene_scale,
        "opacities": t.opacities_lr,
        "sh0": t.sh0_lr,
        "shN": t.shN_lr,
        "mu_t": t.mu_t_lr,
        "quats_l": t.quats_lr,
        "quats_r": t.quats_lr,
        "scales_4d_log": t.scales_lr,
        # appearance: published 4DGS puts every non-DC colour coefficient on feature_lr / 20
        "sh_temporal": t.shN_lr,
        "sh_spacetime": t.shN_lr,
    }

    def dense_adam(name: str, param: torch.nn.Parameter) -> torch.optim.Adam:
        return torch.optim.Adam(
            [{"params": param, "lr": lr_map[name], "name": name}],
            eps=t.optimizer_eps,
            betas=betas,
        )

    if not t.selective_adam:
        return DenseAdamGroup({name: dense_adam(name, param) for name, param in model.splats.items()})

    # Without the CUDA extension the fused kernel falls back to the numerically identical reference.
    stso_cls: type[SpatioTemporalSparseAdam] = SpatioTemporalSparseAdamFused
    if not HAS_CUDA_EXT:
        stso_cls = SpatioTemporalSparseAdamReference
        warnings.warn(
            "fused STSO optimizer requires the CUDA extension; falling back to the reference sparse optimizer.",
            stacklevel=2,
        )
    optimizers: dict[str, SpatioTemporalSparseAdam] = {
        name: stso_cls(
            [{"params": param, "lr": lr_map[name], "name": name}], lr=lr_map[name], eps=t.optimizer_eps, betas=betas
        )
        for name, param in model.splats.items()
    }
    return SpatioTemporalSparseAdamGroup(optimizers)


def build_schedulers(
    optimizers: OptimizerGroupProto,
    training_cfg: TrainingConfig,
) -> list[torch.optim.lr_scheduler.LRScheduler]:
    """Exponential decay for the means LR; other LRs stay constant."""
    opts = optimizers.optimizers
    if "means" not in opts or training_cfg.means_lr <= 0:
        return []
    lr_init = training_cfg.means_lr
    lr_final = training_cfg.means_lr_final
    max_steps = max(training_cfg.max_steps, 1)
    gamma = (lr_final / lr_init) ** (1.0 / max_steps)
    return [torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=gamma)]


def build_bilagrid_optimizer(
    post_processor: BilateralGridPostProcessor,
    training_cfg: TrainingConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Adam and scheduler for the bilateral grid, as in gsplat: 1000-step warmup, then decay to 1%."""
    lr = training_cfg.bilagrid_lr

    optimizer = torch.optim.Adam(
        post_processor.parameters(),
        lr=lr,
        eps=training_cfg.optimizer_eps,
    )

    max_steps = max(training_cfg.max_steps, 1)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=1000),
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01 ** (1.0 / max_steps)),
        ]
    )

    return optimizer, scheduler
