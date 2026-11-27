"""Gradient-driven spatial densification (clone/split/prune) for FourDGaussians.

A dataclass because gsplat's ``Strategy`` base is one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import assert_never

import torch
from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import duplicate
from gsplat.strategy.ops import reset_opa

from anycam4d.console import notice
from anycam4d.enums import SpatialTrigger
from anycam4d.strategy._ops import accumulate_param_spatial_grad
from anycam4d.strategy._ops import accumulate_spatial_grad
from anycam4d.strategy._ops import intrinsic_spatial_scales
from anycam4d.strategy._ops import prune_opacity_and_scale
from anycam4d.strategy._ops import split_spatial
from anycam4d.strategy.events import DEFAULT_EVENTS
from anycam4d.strategy.events import EVENTS_KEY
from anycam4d.strategy.events import init_events
from anycam4d.strategy.events import record


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor
    from torch import nn

    from anycam4d.trainer.optimizers.group import OptimizerGroupProto


def _accumulate_spatial_signal(
    state: dict[str, Any],
    info: dict[str, Any],
    n_gaussians: int,
    trigger: SpatialTrigger,
    packed: bool,
) -> None:
    """Accumulate the active trigger's spatial gradient (``means2d`` for screen space, ``means`` for parameter space).

    Both signals come from ``info``, which the trainer fills per render: ``means_grad`` is that render's
    own means gradient and ``grad_scale`` undoes the ``loss / accum_renders`` scaling.
    """
    grad_scale = info.get("grad_scale", 1.0)
    match trigger:
        case SpatialTrigger.SCREENSPACE:
            accumulate_spatial_grad(state, info, n_gaussians, packed, grad_scale)
        case SpatialTrigger.PARAMSPACE:
            accumulate_param_spatial_grad(state, info, n_gaussians, packed, grad_scale)
        case _:
            assert_never(trigger)


def _zero_spatial_signal(state: dict[str, Any], trigger: SpatialTrigger) -> None:
    """Zero the active trigger's spatial accumulators after a refine (no-op if never accumulated)."""
    match trigger:
        case SpatialTrigger.SCREENSPACE:
            keys = ("grad2d", "count")
        case SpatialTrigger.PARAMSPACE:
            keys = ("grad3d", "count3d")
        case _:
            assert_never(trigger)
    for k in keys:
        if state[k] is not None:
            state[k].zero_()


@dataclass
class DefaultSpatialStrategy(Strategy):
    """Clone/split on spatial gradient; temporal parameters are never mutated.

    Small Gaussians (``max scale <= grow_scale3d * scene_scale``, parameter basis) are cloned, larger
    ones split with :func:`~anycam4d.strategy._ops.split_spatial`. A temporally misplaced Gaussian
    produces no split signal until its spatial error accumulates.

    Args:
        prune_opa: Remove if ``sigmoid(opacity) < prune_opa``; opacity resets go to ``2 * prune_opa``.
        grow_grad2d: Screen-space grow threshold in pixels (``SCREENSPACE`` trigger, EWA only).
        grow_grad3d: Grow threshold on the 3DGRUT signal ``|∂L/∂μ| · distance / 2`` (``PARAMSPACE`` trigger).
        spatial_trigger: Gradient source for growth.
        grow_scale3d: Clone-vs-split boundary as a fraction of ``scene_scale``.
        prune_scale3d: Prune above this fraction of ``scene_scale`` after the first opacity reset.
        refine_start_iter: No densification before this step.
        refine_stop_iter: No densification from this step.
        reset_every: Opacity reset period.
        refine_every: Densification period.
        cap_max: Skip growth at this primitive count (0 disables).
        verbose: Print counts each refine.
        sigma_t_floor: Child temporal scale floor for subclasses whose split shrinks it.
        empty_cache_on_refine: Release cached CUDA memory after each refine.
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_grad3d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    cap_max: int = 0
    verbose: bool = False
    spatial_trigger: SpatialTrigger = SpatialTrigger.PARAMSPACE
    sigma_t_floor: float = 0.0
    empty_cache_on_refine: bool = True

    # Event counter the split kernel's children are recorded under.
    _split_event: ClassVar[str] = "split_spatial"

    def initialize_state(self, scene_scale: float = 1.0, time_extent: float = 1.0) -> dict[str, Any]:  # noqa: ARG002
        """Return initial state dict."""
        return {
            "grad2d": None,
            "count": None,
            "grad3d": None,
            "count3d": None,
            "scene_scale": scene_scale,
            EVENTS_KEY: init_events(DEFAULT_EVENTS),
        }

    def step_pre_backward(
        self,
        params: dict[str, nn.Parameter],  # noqa: ARG002
        optimizers: OptimizerGroupProto,  # noqa: ARG002
        state: dict[str, Any],  # noqa: ARG002
        step: int,  # noqa: ARG002
        info: dict[str, Any],
    ) -> None:
        """Retain ``means2d`` gradients for the screen-space trigger; the parameter-space trigger needs none."""
        if self.spatial_trigger == SpatialTrigger.SCREENSPACE:
            assert "means2d" in info, "means2d is required for spatial_trigger=screenspace but missing."  # noqa: S101
            info["means2d"].retain_grad()

    def accumulate(
        self,
        params: dict[str, nn.Parameter],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
        packed: bool = False,
    ) -> None:
        """Accumulate one render's growth signal; called once per render, before the optimizer step."""
        if step >= self.refine_stop_iter:
            return

        n_gaussians = len(next(iter(params.values())))
        _accumulate_spatial_signal(state, info, n_gaussians, self.spatial_trigger, packed)

    def step_post_backward(
        self,
        params: dict[str, nn.Parameter],
        optimizers: OptimizerGroupProto,
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],  # noqa: ARG002
        packed: bool = False,  # noqa: ARG002
        lr: float = 0.0,  # noqa: ARG002
    ) -> None:
        """Densify/prune on the accumulated signal, and periodically reset opacities."""
        if step >= self.refine_stop_iter:
            return

        if step > self.refine_start_iter and step % self.refine_every == 0:
            at_cap = self.cap_max > 0 and len(params["means"]) >= self.cap_max
            n_dupli, n_split = (0, 0) if at_cap else self._grow_gs(params, optimizers, state)

            n_prune = prune_opacity_and_scale(
                params,
                optimizers,
                state,
                opa_thr=self.prune_opa,
                spatial_scale_thr=self.prune_scale3d,
                scene_scale=state["scene_scale"],
                after_first_reset=step > self.reset_every,
            )

            record(state, cloned=n_dupli, pruned=n_prune, **{self._split_event: n_split})

            if self.verbose:
                cap_str = " (cap)" if at_cap else ""
                notice(
                    f"Step {step}: {n_dupli} cloned, {n_split} split, {n_prune} pruned{cap_str} → {len(params['means'])} Gaussians."
                )

            self._zero_grow_signal(state)

            if self.empty_cache_on_refine:
                torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers.optimizers, state=state, value=self.prune_opa * 2.0)
            record(state, opacity_resets=1)

    def _spatial_grow_signal(self, state: dict[str, Any]) -> tuple[Float[Tensor, "N"] | None, float]:
        """Mean accumulated grow signal (``None`` if nothing accumulated) and its threshold for the active trigger."""
        if self.spatial_trigger == SpatialTrigger.SCREENSPACE:
            grad_key, count_key, thr = "grad2d", "count", self.grow_grad2d
        else:
            grad_key, count_key, thr = "grad3d", "count3d", self.grow_grad3d
        if state[grad_key] is None:
            return None, thr
        return state[grad_key] / state[count_key].clamp_min(1), thr

    @torch.no_grad()
    def _grow_gs(
        self,
        params: dict[str, nn.Parameter],
        optimizers: OptimizerGroupProto,
        state: dict[str, Any],
    ) -> tuple[int, int]:
        """Clone and split Gaussians based on spatial gradient."""
        opt = optimizers.optimizers
        grads, grow_thr = self._spatial_grow_signal(state)
        if grads is None:
            return 0, 0

        s_xyz_max = intrinsic_spatial_scales(params).max(dim=-1).values

        # Inclusive, as in the published densifier (exact hits occur).
        is_grad_high = grads >= grow_thr
        is_dupli = is_grad_high & (s_xyz_max <= self.grow_scale3d * state["scene_scale"])
        is_split = is_grad_high & (s_xyz_max > self.grow_scale3d * state["scene_scale"])

        n_dupli = int(is_dupli.sum().item())
        n_split = int(is_split.sum().item())
        device = grads.device

        if n_dupli > 0:
            duplicate(params=params, optimizers=opt, state=state, mask=is_dupli)

        is_split_padded = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
        if n_split > 0:
            self._spatial_split(params, optimizers, state, is_split_padded)

        return n_dupli, n_split

    def _zero_grow_signal(self, state: dict[str, Any]) -> None:
        """Zero the growth accumulators after a refine."""
        _zero_spatial_signal(state, self.spatial_trigger)

    def _spatial_split(
        self,
        params: dict[str, nn.Parameter],
        optimizers: OptimizerGroupProto,
        state: dict[str, Any],
        mask: Bool[Tensor, "N"],
    ) -> None:
        """Split kernel seam; ``OGFourDGSStrategy`` overrides it with the coupled 4D split."""
        split_spatial(params, optimizers, state, mask)
