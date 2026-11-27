"""Training config dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from anycam4d.enums import LpipsNet
from anycam4d.enums import SamplerType


if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class TrainingConfig:
    """Training schedule, learning rates, and sampler settings."""

    max_steps: int = 30_000
    # Seed for all RNGs. Runs are not bit-reproducible (atomicAdd), so vary it across replicates.
    seed: int = 42
    # How many cameras one optimizer step renders; learning rates do not scale with it. -1 = take it from the
    # data (camera count under `frame_aligned`, 1 otherwise), resolved in scripts/train.py.
    batch_size: int = 1
    # Renders accumulated per optimizer step: N renders, `loss / N`, one step (published 4DGS's batch).
    # Effective batch = batch_size * accum_renders. "step" stays the optimizer step every schedule indexes on.
    accum_renders: int = 1
    sampler: SamplerType = SamplerType.UNIFORM

    # Spatial LRs (match gsplat simple_trainer defaults)
    means_lr: float = 1.6e-4
    quats_lr: float = 1e-3
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20.0  # 1.25e-4

    # Temporal LRs (conservative: 0.1x spatial)
    mu_t_lr: float = 1.6e-5

    # LR decay
    means_lr_final: float = 1.6e-6

    # Bilateral grid LR (active only when rendering.post_processor = bilateral_grid)
    bilagrid_lr: float = 2e-3

    # SSIM schedule
    ssim_start_step: int = 500

    # Evaluation
    eval_every: int = 1000
    lpips_net: LpipsNet = LpipsNet.ALEX

    # Logging
    log_every: int = 10
    save_every: int = 0
    save_best: bool = True
    best_metric: str = "psnr"
    save_images: bool = False

    # SH curriculum
    sh_degree_interval: int = 1000

    # Adam optimizer
    optimizer_eps: float = 1e-15
    optimizer_betas: tuple[float, float] = (0.9, 0.999)

    # Loss weights
    ssim_lambda: float = 0.2
    mask_bg_alpha_lambda: float = 0.2
    # L1 spatial-scale regularizer weight.
    scale_reg_lambda: float = 0.0
    # Opacity regularizer weight: L1 on sigmoid(alpha) * w_t(tau), which manufactures MCMC's dead set.
    # 0 makes an MCMC run stall (warned at startup).
    opacity_reg_lambda: float = 0.0
    # Stop the gradient at w_t so opacity_reg acts on opacities only; with a live gate it also
    # pushes geometry and can switch itself off by narrowing the temporal width.
    opacity_reg_detach_gate: bool = False

    # Bilateral grid total variation regularization weight (active only when rendering.post_processor = bilateral_grid)
    bilagrid_tv_lambda: float = 10.0

    # Selective Adam: steps only the primitives visible in the batch; off is dense Adam.
    selective_adam: bool = False

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> TrainingConfig:
        """Create TrainingConfig from OmegaConf DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        return TrainingConfig(
            max_steps=raw["max_steps"],
            seed=raw["seed"],
            batch_size=raw["batch_size"],
            accum_renders=raw["accum_renders"],
            sampler=SamplerType(raw["sampler"]),
            means_lr=raw["means_lr"],
            quats_lr=raw["quats_lr"],
            scales_lr=raw["scales_lr"],
            opacities_lr=raw["opacities_lr"],
            sh0_lr=raw["sh0_lr"],
            shN_lr=raw["shN_lr"],
            mu_t_lr=raw["mu_t_lr"],
            means_lr_final=raw["means_lr_final"],
            ssim_start_step=raw["ssim_start_step"],
            log_every=raw["log_every"],
            eval_every=raw["eval_every"],
            lpips_net=LpipsNet(raw["lpips_net"]),
            save_every=raw["save_every"],
            save_best=raw["save_best"],
            best_metric=raw["best_metric"],
            save_images=raw["save_images"],
            sh_degree_interval=raw["sh_degree_interval"],
            optimizer_eps=raw["optimizer_eps"],
            optimizer_betas=tuple(raw["optimizer_betas"]),
            ssim_lambda=raw["ssim_lambda"],
            mask_bg_alpha_lambda=raw["mask_bg_alpha_lambda"],
            scale_reg_lambda=raw["scale_reg_lambda"],
            opacity_reg_lambda=raw["opacity_reg_lambda"],
            opacity_reg_detach_gate=raw["opacity_reg_detach_gate"],
            bilagrid_tv_lambda=raw["bilagrid_tv_lambda"],
            bilagrid_lr=raw["bilagrid_lr"],
            # .get: checkpoint snapshots predate this field (eval reads them; it never builds an optimizer).
            selective_adam=raw.get("selective_adam", False),
        )
