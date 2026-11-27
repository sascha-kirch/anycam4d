"""MCMC densification for 4D Gaussians (Kheradmand et al., 2024), spatial moves only.

Temporal noise is not injected: ``cov_4d[3, 3]`` noise would exceed the whole ``mu_t`` range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

import torch
from gsplat.strategy.base import Strategy

from anycam4d.console import notice
from anycam4d.enums import ParentScore
from anycam4d.strategy._ops import accumulate_param_spatial_grad
from anycam4d.strategy._ops import inject_noise_spatial
from anycam4d.strategy._ops import percentile_rank
from anycam4d.strategy._ops import primitive_speed
from anycam4d.strategy._ops import relocate_spatial
from anycam4d.strategy._ops import sample_add_spatial
from anycam4d.strategy.events import EVENTS_KEY
from anycam4d.strategy.events import MCMC_EVENTS
from anycam4d.strategy.events import init_events
from anycam4d.strategy.events import record


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor
    from torch import nn

    from anycam4d.trainer.optimizers.group import OptimizerGroupProto


# Under the motion score the fastest parent weighs (1 + MOTION_BOOST)x the slowest at equal error; static content keeps
# its error weight, so it still densifies.
MOTION_BOOST = 3.0


def _build_binoms(n_max: int = 51) -> Float[Tensor, "n_max n_max"]:
    """Pre-compute binomial coefficients table for ``compute_relocation``."""
    binoms = torch.zeros((n_max, n_max))
    for n in range(n_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)
    return binoms


@dataclass
class MCMCSpatialStrategy(Strategy):
    """MCMC densification (relocate, sample-add, Langevin noise) with spatial-only moves.

    Ports ``gsplat.MCMCStrategy``: fitness is base opacity, noise perturbs ``means`` only, temporal
    parameters are inherited verbatim. Never prunes; dead primitives are relocated onto live parents.

    Args:
        cap_max: Primitive cap; sample-add grows the population by growth_rate per refine until reached.
        noise_lr: Noise amplitude multiplier; the effective scaler is ``lr * noise_lr``, annealing with the means LR.
        refine_start_iter: Relocate/sample-add start after this step (noise runs from step 0).
        refine_stop_iter: Relocate/sample-add stop at this step.
        noise_injection_stop_iter: Noise stops at this step; ``-1`` couples it to ``refine_stop_iter``.
        refine_every: Refine period.
        min_opacity: Dead threshold ``sigmoid(opacity) <= min_opacity``.
        growth_rate: Population growth per refine until ``cap_max``.
        birth_opacity_factor: Children are born at ``min_opacity * factor``; 1.0 (published) makes a
            clamped child dead again at the next refine.
        temporal_partition: Relocated and added copies split their parent's temporal footprint instead of
            copying it.
        parent_score: What relocation and growth parents are sampled by (see ``enums.ParentScore``).
        verbose: Print counts each refine.
        empty_cache_on_refine: Release cached CUDA memory after each refine.
    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    noise_injection_stop_iter: int = 20_000
    refine_every: int = 100
    min_opacity: float = 0.005
    birth_opacity_factor: float = 1.0
    growth_rate: float = 0.05
    temporal_partition: bool = False
    parent_score: ParentScore = ParentScore.OPACITY
    verbose: bool = False
    empty_cache_on_refine: bool = True
    # Constant table, kept off ``state`` because the densification ops resize every tensor there.
    _binoms: Float[Tensor, "n_max n_max"] | None = field(default=None, init=False, repr=False, compare=False)

    def initialize_state(self, scene_scale: float = 1.0, time_extent: float = 1.0) -> dict[str, Any]:  # noqa: ARG002
        """Build the constant binomial table; the state carries the event counters and the parent-score gradient."""
        self._binoms = _build_binoms()
        return {EVENTS_KEY: init_events(MCMC_EVENTS), "grad3d": None, "count3d": None}

    def check_sanity(
        self,
        params: dict[str, nn.Parameter],
        optimizers: OptimizerGroupProto,
    ) -> None:
        """Check required parameter keys are present."""
        super().check_sanity(params, optimizers.optimizers)
        for key in ["means", "mu_t", "scales_4d_log", "quats_l", "quats_r", "opacities"]:
            assert key in params, f"{key} is required in params but missing."  # noqa: S101

    def accumulate(
        self,
        params: dict[str, nn.Parameter],
        state: dict[str, Any],
        step: int,  # noqa: ARG002
        info: dict[str, Any],
        packed: bool = False,
    ) -> None:
        """Accumulate the per-view image gradient the motion score reads; a no-op under opacity."""
        if self.parent_score != ParentScore.OPACITY:
            accumulate_param_spatial_grad(state, info, len(params["means"]), packed, info.get("grad_scale", 1.0))

    def step_post_backward(
        self,
        params: dict[str, nn.Parameter],
        optimizers: OptimizerGroupProto,
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],  # noqa: ARG002
        packed: bool = False,  # noqa: ARG002
        lr: float = 0.0,
    ) -> None:
        """Run MCMC moves: relocate dead, add new, inject noise."""
        self._binoms = self._binoms.to(params["means"].device)
        binoms = self._binoms

        if self.refine_start_iter < step < self.refine_stop_iter and step % self.refine_every == 0:
            # Relocation zeroes the gradient rows it touches, so growth reuses the motion weights drawn before it.
            weights = self.parent_weights(params, state)
            n_relocated, dead_mask = self._relocate_gs(params, optimizers, state, binoms, weights)
            if self.parent_score == ParentScore.OPACITY:
                weights = self.parent_weights(params, state)
            else:
                weights = weights * ~dead_mask
            n_new = self._add_new_gs(params, optimizers, state, binoms, weights)

            record(state, relocated=n_relocated, added=n_new, refine_steps=1)
            for key in ("grad3d", "count3d"):
                if state[key] is not None:
                    state[key].zero_()

            if self.verbose:
                notice(f"Step {step}: relocated {n_relocated}, added {n_new}. Now {len(params['means'])} Gaussians.")

            if self.empty_cache_on_refine:
                torch.cuda.empty_cache()

        noise_stop = self.noise_injection_stop_iter if self.noise_injection_stop_iter >= 0 else float("inf")

        if step < noise_stop:
            inject_noise_spatial(params, state, scaler=lr * self.noise_lr)

    def fitness(
        self,
        params: dict[str, nn.Parameter],
        state: dict[str, Any],  # noqa: ARG002
    ) -> Float[Tensor, "N"]:
        """Per-primitive worth: base opacity, the quantity this family's dead test reads."""
        return torch.sigmoid(params["opacities"].flatten())

    def dead_threshold(self) -> float:
        """The cut applied to :meth:`fitness`."""
        return self.min_opacity

    def dead_mask(self, score: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        """Primitives relocation treats as dead, thresholding :meth:`fitness` at :meth:`dead_threshold`."""
        return score <= self.dead_threshold()

    def parent_weights(self, params: dict[str, nn.Parameter], state: dict[str, Any]) -> Float[Tensor, "N"]:
        """Parent-sampling weights."""
        opacity = torch.sigmoid(params["opacities"].flatten())
        speed = primitive_speed(params)
        if self.parent_score == ParentScore.OPACITY:
            weights = opacity
        else:
            grad = state["grad3d"] / state["count3d"].clamp_min(1) if state["grad3d"] is not None else 0 * opacity
            error = 0.5 * percentile_rank(opacity) + 0.5 * percentile_rank(grad)
            weights = error * (1.0 + MOTION_BOOST * percentile_rank(speed))
            # Rank terms give a dead primitive weight; opacity alone never had to exclude it.
            weights = weights * (opacity > self.min_opacity)
        return weights

    def birth_opacity(self) -> float:
        """Opacity relocated and sampled children are born with: ``min_opacity * birth_opacity_factor``."""
        return self.min_opacity * self.birth_opacity_factor

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: dict[str, nn.Parameter],
        optimizers: OptimizerGroupProto,
        state: dict[str, Any],
        binoms: Float[Tensor, "n_max n_max"],
        weights: Float[Tensor, "N"],
    ) -> tuple[int, Bool[Tensor, "N"]]:
        """Relocate Gaussians with opacity <= min_opacity to live parents; returns the count and the dead mask."""
        opacity = self.fitness(params, state)
        dead_mask = self.dead_mask(opacity)
        n_gs = int(dead_mask.sum().item())

        if n_gs > 0:
            relocate_spatial(
                params,
                optimizers,
                state,
                dead_mask,
                binoms,
                self.birth_opacity(),
                self.temporal_partition,
                weights,
            )

        return n_gs, dead_mask

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: dict[str, nn.Parameter],
        optimizers: OptimizerGroupProto,
        state: dict[str, Any],
        binoms: Float[Tensor, "n_max n_max"],
        weights: Float[Tensor, "N"],
    ) -> int:
        """Append up to ``growth_rate`` more Gaussians, capped at cap_max."""
        current_n = len(params["means"])
        n_gs = max(0, min(self.cap_max, int((1.0 + self.growth_rate) * current_n)) - current_n)

        if n_gs > 0:
            sample_add_spatial(
                params,
                optimizers,
                state,
                n_gs,
                binoms,
                self.birth_opacity(),
                self.temporal_partition,
                weights,
            )

        return n_gs
