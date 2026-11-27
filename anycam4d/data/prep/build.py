"""Compose per-view prep stages into one transform."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    from anycam4d.data.prep.sample import CalibratedView
    from anycam4d.data.prep.stream import StageProto


def compose(stages: Iterable[StageProto]) -> Callable[[CalibratedView], CalibratedView]:
    """Compose stages left-to-right; ``compose([])`` is the identity transform.

    Args:
        stages: Ordered pivot -> pivot transforms.

    Returns:
        A single callable that applies every stage in order.
    """

    def _apply(view: CalibratedView) -> CalibratedView:
        for stage in stages:
            view = stage(view)
        return view

    return _apply
