"""Cumulative densification-event counters on a strategy's state dict.

The population size is a net number: cloning 40 000 primitives while pruning 40 000 looks like nothing happened.
These counters record the gross events. They are cumulative because the refine and logging cadences are
independent; the per-refine rate is recovered by differencing. Plain ints, nothing on device.
"""

from __future__ import annotations

from typing import Any


# Key under which the counter dict lives on a strategy's state dict.
EVENTS_KEY = "refine_events"

# Gradient-driven families. ``split_spatial`` and ``split_full_4d`` are separated because only the
# coupled kernel mutates mu_t and scale_t.
DEFAULT_EVENTS = (
    "cloned",
    "split_spatial",
    "split_full_4d",
    "pruned",
    "opacity_resets",
)

# Population-driven family. MCMC never prunes; it recycles through relocation and grows by sampling.
MCMC_EVENTS = (
    "relocated",
    "added",
    "refine_steps",
)


def init_events(names: tuple[str, ...]) -> dict[str, int]:
    """Return a zeroed counter dict for the given event names."""
    return dict.fromkeys(names, 0)


def record(state: dict[str, Any], **counts: int) -> None:
    """Add ``counts`` to the state's cumulative event counters.

    A no-op when the state carries no counter dict, so a strategy that has not opted in — or a
    caller holding a state built before this module existed — keeps working untouched.
    """
    events = state.get(EVENTS_KEY)
    if events is None:
        return
    for name, value in counts.items():
        events[name] = events.get(name, 0) + int(value)
