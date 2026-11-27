"""Control strategy: no densification, the population keeps its initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gsplat.strategy.base import Strategy


@dataclass
class NullStrategy(Strategy):
    """Strategy that does nothing."""

    def initialize_state(self, *args: object, **kwargs: object) -> dict[str, Any]:  # noqa: ARG002 — no-op strategy ignores all inputs
        """Return the initial running state dict."""
        return {}

    def step_pre_backward(self, *args: object, **kwargs: object) -> None:
        """Skip this step."""

    def accumulate(self, *args: object, **kwargs: object) -> None:
        """Skip this render."""

    def step_post_backward(self, *args: object, **kwargs: object) -> None:
        """Skip this step."""
