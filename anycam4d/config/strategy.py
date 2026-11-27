"""Densification strategy config dataclass."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from anycam4d.enums import ParentScore
from anycam4d.enums import SpatialTrigger
from anycam4d.enums import StrategyType


if TYPE_CHECKING:
    from omegaconf import DictConfig


_MCMC_TYPES = {StrategyType.MCMC_SPATIAL}
_DEFAULT_TYPES = {StrategyType.DEFAULT_SPATIAL, StrategyType.OG_4DGS}


@dataclass
class StrategyConfig:
    """Densification strategy configuration.

    Three families share this one config: the gradient-driven ``default_spatial`` / ``og_4dgs``,
    the population-driven ``mcmc_spatial``, and the ``null_4dgs`` control. A field consumed by only
    one of them raises or warns on the others rather than being silently ignored.
    """

    type: StrategyType = StrategyType.MCMC_SPATIAL

    # Spatial densification
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_opa: float = 0.005
    prune_scale3d: float = 0.1
    # --- Densification schedule ------------------------------------------------------------
    # Absolute step counts, validated against training.max_steps by StrategyFactory.create.
    refine_start_iter: int = 500
    refine_stop_iter: int = 22_500
    # Step at which MCMC noise injection stops (-1: at refine_stop_iter). Ignored by other families.
    noise_injection_stop_iter: int = 22_500
    refine_every: int = 100
    reset_every: int = 3000

    # Primitive cap for every family; 0 disables it for the gradient-driven ones.
    cap_max: int = 0
    noise_lr: float = 5e5
    # MCMC dead threshold on sigmoid(opacity); children are born at min_opacity * birth_opacity_factor.
    min_opacity: float = 0.005
    # Multiple of ``min_opacity`` a relocated / sampled child is born at. 1.0 = published behaviour,
    # where birth floor and dead threshold coincide and a clamped child is dead on arrival. See
    # ``strategy/mcmc.py:MCMCSpatialStrategy.birth_opacity_factor``.
    birth_opacity_factor: float = 1.0
    # MCMC population growth per refine until cap_max (gsplat: 5 %).
    growth_rate: float = 0.05
    # MCMC: relocated / added copies split the parent's temporal footprint instead of copying it.
    temporal_partition: bool = False
    # MCMC: what relocation and growth parents are sampled by (enums.ParentScore).
    parent_score: ParentScore = ParentScore.OPACITY

    # Spatial grow trigger: `paramspace` (|means.grad|, any backend) or `screenspace` (EWA only).
    spatial_trigger: SpatialTrigger = SpatialTrigger.PARAMSPACE

    # Floor on the intrinsic temporal scale exp(scales_4d_log[:, 3]) at split and init; -1.0 = one frame.
    sigma_t_floor: float = -1.0

    # Parameter-space grow threshold on |∂L/∂μ| · camera distance / 2 (3DGRUT; its value is the 3DGS one).
    grad3d_threshold: float = 0.0002

    # Release cached CUDA memory after each refine; disable when not VRAM-bound.
    empty_cache_on_refine: bool = True

    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate coupled fields and warn on wrong-family settings.

        Fails loud on a contradictory schedule and a meaningless cap, and emits a single grouped
        warning when obviously-wrong-family fields carry non-default values.
        """
        self._validate_schedule()
        if self.type in _MCMC_TYPES and self.cap_max < 1:
            raise ValueError(
                f"MCMC strategies require cap_max >= 1 (got {self.cap_max}); set an explicit "
                f"population cap in the strategy config."
            )
        self._warn_wrong_family()

    def _validate_schedule(self) -> None:
        """Reject a refinement window that is empty or negative.

        Run-length agreement is not checkable here -- ``max_steps`` lives in the training config --
        so it is enforced in :meth:`validate_against_run`, which the factory calls.
        """
        if self.refine_start_iter < 0:
            raise ValueError(f"refine_start_iter must be >= 0 (got {self.refine_start_iter}).")
        if self.refine_stop_iter <= self.refine_start_iter:
            raise ValueError(
                f"refine_stop_iter ({self.refine_stop_iter}) must be greater than refine_start_iter "
                f"({self.refine_start_iter}); the refinement window would be empty and densification "
                f"would never fire."
            )
        if self.type in _MCMC_TYPES and self.noise_injection_stop_iter > self.refine_stop_iter:
            raise ValueError(
                f"noise_injection_stop_iter ({self.noise_injection_stop_iter}) must not exceed "
                f"refine_stop_iter ({self.refine_stop_iter}); noise would be injected into a "
                f"population densification has already frozen."
            )

    def validate_against_run(self, max_steps: int) -> None:
        """Raise if the densification schedule does not fit inside ``max_steps`` (a stop past the end silently removes the settling phase)."""
        if self.refine_start_iter >= max_steps:
            raise ValueError(
                f"strategy.refine_start_iter ({self.refine_start_iter:,}) is at or past "
                f"training.max_steps ({max_steps:,}), so densification never starts. Lower it "
                f"(short probe runs want ~50) or raise max_steps."
            )
        if self.type in _MCMC_TYPES and self.noise_injection_stop_iter > max_steps:
            raise ValueError(
                f"strategy.noise_injection_stop_iter ({self.noise_injection_stop_iter:,}) is past "
                f"training.max_steps ({max_steps:,}), so noise injection never stops."
            )
        if self.refine_stop_iter > max_steps:
            raise ValueError(
                f"strategy.refine_stop_iter ({self.refine_stop_iter:,}) is past training.max_steps "
                f"({max_steps:,}), so densification never stops and runs to the last step -- the "
                f"settling phase the schedule exists to create is silently deleted. Set "
                f"refine_stop_iter <= {max_steps:,} (use exactly {max_steps:,} to densify throughout)."
            )

    def _warn_wrong_family(self) -> None:
        """Warn about non-default fields that the selected family ignores."""
        # (field name, current value, dataclass default) tuples for the family that ignores them.
        # Only non-default values are flaggable — a plain dataclass has no set-vs-defaulted signal.
        if self.type in _DEFAULT_TYPES:
            candidates = [
                ("noise_lr", self.noise_lr, 5e5),
                ("min_opacity", self.min_opacity, 0.005),
            ]
        elif self.type in _MCMC_TYPES:
            candidates = [("grow_grad2d", self.grow_grad2d, 0.0002), ("grow_scale3d", self.grow_scale3d, 0.01)]
        else:
            candidates = []
        ignored = [name for name, value, default in candidates if value != default]
        if ignored:
            warnings.warn(
                f"StrategyConfig type={self.type.value} ignores these set-but-unused fields: {', '.join(ignored)}",
                stacklevel=2,
            )

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> StrategyConfig:
        """Create StrategyConfig from OmegaConf DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        return StrategyConfig(
            type=StrategyType(raw["type"]),
            grow_grad2d=raw.get("grow_grad2d", 0.0002),
            grow_scale3d=raw.get("grow_scale3d", 0.01),
            prune_opa=raw.get("prune_opa", 0.005),
            prune_scale3d=raw.get("prune_scale3d", 0.1),
            refine_start_iter=raw.get("refine_start_iter", 500),
            refine_stop_iter=raw.get("refine_stop_iter", 22_500),
            noise_injection_stop_iter=raw.get("noise_injection_stop_iter", 22_500),
            refine_every=raw.get("refine_every", 100),
            reset_every=raw.get("reset_every", 3000),
            cap_max=raw.get("cap_max", 0),
            noise_lr=raw.get("noise_lr", 5e5),
            min_opacity=raw.get("min_opacity", 0.005),
            birth_opacity_factor=raw.get("birth_opacity_factor", 1.0),
            growth_rate=raw.get("growth_rate", 0.05),
            temporal_partition=raw.get("temporal_partition", False),
            parent_score=ParentScore(raw.get("parent_score", "opacity")),
            spatial_trigger=SpatialTrigger(raw.get("spatial_trigger", "paramspace")),
            sigma_t_floor=raw.get("sigma_t_floor", -1.0),
            grad3d_threshold=raw.get("grad3d_threshold", 0.0002),
            empty_cache_on_refine=raw.get("empty_cache_on_refine", True),
            verbose=raw.get("verbose", False),
        )
