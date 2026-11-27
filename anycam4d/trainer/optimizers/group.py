"""Optimizer groups: dense Adam or sparse spatio-temporal Adam behind one interface.

Densification ops resize parameters and moments themselves through ``optimizers``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable


if TYPE_CHECKING:
    from jaxtyping import Bool
    from torch import Tensor
    from torch.optim import Optimizer

    from anycam4d.trainer.optimizers.stso import SpatioTemporalSparseAdam


@runtime_checkable
class OptimizerGroupProto(Protocol):
    """Common interface for the dense and sparse spatio-temporal optimizer paths."""

    optimizers: dict[str, Optimizer]  # the gsplat densification ops operate on this dict

    def step(self, visibility: Bool[Tensor, "N"] | None = None) -> None:
        """Run one optimizer step (the sparse group consumes the mask; the dense group drops it)."""
        ...

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero all parameter gradients."""
        ...


class DenseAdamGroup(OptimizerGroupProto):
    """Wraps ``dict[str, torch.optim.Adam]``; the activity mask is ignored."""

    def __init__(self, optimizers: dict[str, Optimizer]) -> None:
        """Store the per-parameter Adam optimizers built by ``build_optimizers``."""
        self.optimizers = optimizers

    def step(self, visibility: Bool[Tensor, "N"] | None = None) -> None:  # noqa: ARG002 — mask intentionally dropped
        """Dense Adam step over every parameter (the activity mask does not apply)."""
        for opt in self.optimizers.values():
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero all parameter gradients."""
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)


class SpatioTemporalSparseAdamGroup(OptimizerGroupProto):
    """Per-parameter sparse optimizers (selective Adam)."""

    def __init__(self, optimizers: dict[str, SpatioTemporalSparseAdam]) -> None:
        """Store the per-parameter optimizers built by ``build_optimizers``."""
        self.optimizers = optimizers

    def step(self, visibility: Bool[Tensor, "N"] | None = None) -> None:
        """Step every sparse optimizer on the active set."""
        assert visibility is not None, "a visibility mask is required"  # noqa: S101
        for opt in self.optimizers.values():
            opt.step(visibility)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero all parameter gradients."""
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)
