"""Spatio-temporal sparse Adam: update only primitives active at the rendered timestamp (Selective Adam).

The active set is ``radii > 0``; under the temporal prefilter that already excludes temporally gated
primitives. Inactive rows keep ``θ``, ``m`` and ``v`` frozen. No bias correction::

    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g^2
    theta -= lr * m / (sqrt(v) + eps)          # active rows only
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from anycam4d.cuda.ext import stso_adam_update


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from jaxtyping import Shaped
    from torch import Tensor


class SpatioTemporalSparseAdam(torch.optim.Adam):
    """Shared contract: one tensor per group; the group passes the activity mask each step."""

    @torch.no_grad()
    def step(self, visibility: Bool[Tensor, "N"]) -> None:  # type: ignore[override]
        """Apply one sparse update (subclasses)."""
        raise NotImplementedError

    def _lazy_state(self, p: Float[Tensor, "N ..."]) -> tuple[Float[Tensor, "N ..."], Float[Tensor, "N ..."]]:
        """Return ``(exp_avg, exp_avg_sq)`` for ``p``, initializing both to zeros on first touch."""
        state = self.state[p]
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        return state["exp_avg"], state["exp_avg_sq"]

    @staticmethod
    def _broadcast(vec: Shaped[Tensor, "N"], p: Float[Tensor, "N ..."], n: int) -> Shaped[Tensor, "N ..."]:
        """Broadcast an ``[N]`` mask over a parameter of shape ``[N, ...]``."""
        return vec.view(n, *([1] * (p.dim() - 1))).expand_as(p)


class SpatioTemporalSparseAdamReference(SpatioTemporalSparseAdam):
    """Pure-PyTorch oracle for the fused kernel, and its fallback without the CUDA extension."""

    @torch.no_grad()
    def step(self, visibility: Bool[Tensor, "N"]) -> None:  # type: ignore[override]
        """Advance the active rows by one Adam step; leave inactive rows' ``m``, ``v``, ``theta`` frozen."""
        n = visibility.numel()
        for group in self.param_groups:
            params = group["params"]
            assert len(params) == 1, "one tensor per parameter group (gsplat layout)"  # noqa: S101
            p = params[0]
            if p.grad is None:
                continue

            m, v = self._lazy_state(p)
            g = p.grad
            beta1, beta2 = group["betas"]
            active = self._broadcast(visibility, p, n)
            m_new = beta1 * m + (1.0 - beta1) * g
            v_new = beta2 * v + (1.0 - beta2) * g * g

            upd = group["lr"] * m_new / (v_new.sqrt() + group["eps"])
            p.add_(torch.where(active, -upd, torch.zeros_like(upd)))
            m.copy_(torch.where(active, m_new, m))
            v.copy_(torch.where(active, v_new, v))


class SpatioTemporalSparseAdamFused(SpatioTemporalSparseAdam):
    """Fused CUDA ``stso_adam_update``; validated against the reference."""

    @torch.no_grad()
    def step(self, visibility: Bool[Tensor, "N"]) -> None:  # type: ignore[override]
        """Advance the active rows via the fused kernel; leave inactive rows' ``m``, ``v``, ``theta`` frozen."""
        for group in self.param_groups:
            params = group["params"]
            assert len(params) == 1, "one tensor per parameter group (gsplat layout)"  # noqa: S101
            p = params[0]
            if p.grad is None:
                continue

            m, v = self._lazy_state(p)
            beta1, beta2 = group["betas"]
            assert visibility.device == p.device, "visibility must be on the param device"  # noqa: S101
            assert visibility.dtype == torch.bool, "visibility must be bool"  # noqa: S101
            stso_adam_update(p, p.grad, m, v, visibility, group["lr"], beta1, beta2, group["eps"])
