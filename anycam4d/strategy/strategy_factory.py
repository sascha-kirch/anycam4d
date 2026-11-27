"""Factory for creating densification strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anycam4d.enums import StrategyType
from anycam4d.renderer.render_registry import check_spatial_trigger_capability
from anycam4d.strategy.coupled import OGFourDGSStrategy
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.strategy.mcmc import MCMCSpatialStrategy
from anycam4d.strategy.null_strategy import NullStrategy


if TYPE_CHECKING:
    from gsplat.strategy.base import Strategy

    from anycam4d.config.strategy import StrategyConfig
    from anycam4d.enums import BackendType


# Spatial-only-selection family: same kwargs + gate, distinguished only by the split kernel their
# class bakes in (DefaultSpatial = spatial marginal; OG_4DGS = published full-4D split).
_SPATIAL_FAMILY = {
    StrategyType.DEFAULT_SPATIAL: DefaultSpatialStrategy,
    StrategyType.OG_4DGS: OGFourDGSStrategy,
}


class StrategyFactory:
    """Creates densification strategies from config."""

    @staticmethod
    def create(cfg: StrategyConfig, backend: BackendType, max_steps: int) -> Strategy:
        """Build a densification strategy from config.

        Args:
            cfg: Strategy configuration.
            backend: Active render backend. Used to reject a screen-space spatial trigger on a
                backend that emits no differentiable ``means2d`` (the Default family only).
            max_steps: Run length, validated against the densification schedule. The schedule is
                configured in absolute steps and ``max_steps`` is an independent knob, so this is
                the one place the two are checked to agree -- the check whose absence let 447 runs
                densify to their last step.

        Returns:
            Configured strategy instance.

        Raises:
            ValueError: If the refinement window does not fit inside a run of ``max_steps``.
        """
        cfg.validate_against_run(max_steps)
        # DEFAULT_SPATIAL and OG_4DGS share the spatial-only selection path, the same gate, and the
        # same kwargs — they differ only in the split kernel (baked into the class). OG_4DGS's kernel
        # is the coupled full-4D one that mutates mu_t/scale_t without temporal evidence.
        if cfg.type in _SPATIAL_FAMILY:
            check_spatial_trigger_capability(backend, cfg.spatial_trigger)
            return _SPATIAL_FAMILY[cfg.type](
                grow_grad2d=cfg.grow_grad2d,
                grow_grad3d=cfg.grad3d_threshold,
                spatial_trigger=cfg.spatial_trigger,
                grow_scale3d=cfg.grow_scale3d,
                prune_opa=cfg.prune_opa,
                prune_scale3d=cfg.prune_scale3d,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                refine_every=cfg.refine_every,
                reset_every=cfg.reset_every,
                cap_max=cfg.cap_max,
                verbose=cfg.verbose,
                sigma_t_floor=cfg.sigma_t_floor,
                empty_cache_on_refine=cfg.empty_cache_on_refine,
            )

        if cfg.type == StrategyType.MCMC_SPATIAL:
            return MCMCSpatialStrategy(
                cap_max=cfg.cap_max,  # validator guarantees MCMC cap_max >= 1
                noise_lr=cfg.noise_lr,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                noise_injection_stop_iter=cfg.noise_injection_stop_iter,
                refine_every=cfg.refine_every,
                min_opacity=cfg.min_opacity,
                birth_opacity_factor=cfg.birth_opacity_factor,
                growth_rate=cfg.growth_rate,
                temporal_partition=cfg.temporal_partition,
                parent_score=cfg.parent_score,
                verbose=cfg.verbose,
                empty_cache_on_refine=cfg.empty_cache_on_refine,
            )

        if cfg.type == StrategyType.NULL_4DGS:
            return NullStrategy()

        raise ValueError(f"Unknown strategy type: {cfg.type}")
