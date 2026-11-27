"""Headless unit tests for TemporalSchedule (frame counts + τ(i), no viser server)."""

import pytest

from anycam4d.viewer.temporal_schedule import TemporalSchedule


_GRID = [
    ("Manual", 100, 1.0, 30.0, 4.0, False, -0.5),
    ("Manual", 60, 2.0, 24.0, 2.5, False, 0.1),
    ("Loop", 100, 1.0, 30.0, 4.0, False, -0.5),
    ("Loop", 100, 2.5, 30.0, 4.0, False, -0.5),
    ("Loop", 100, 0.25, 30.0, 4.0, True, -0.5),
    ("Loop", 7, 3.0, 12.0, 1.0, True, -0.5),
    ("Freeze", 100, 1.0, 30.0, 4.0, False, 0.25),
]


def test_loop_endpoints() -> None:
    """A single Loop sweep starts at -0.5 (i=0) and reaches 0.5 (i=n-1)."""
    sched = TemporalSchedule("Loop", 30.0, 4.0, 100, loop_count=1.0)
    n = sched.n_frames(minimum=1)
    assert sched.tau(0, 0.0) == pytest.approx(-0.5)
    assert sched.tau(n - 1, 0.0) == pytest.approx(0.5)


def test_n_frames_minimum_floor() -> None:
    """The minimum floor differs between display (1) and spline viz (2)."""
    sched = TemporalSchedule("Loop", 30.0, 0.0, 1, loop_count=0.25)
    assert sched.n_frames(minimum=1) == 1
    assert sched.n_frames(minimum=2) == 2


@pytest.mark.parametrize(("mode", "num_ts", "loop_count", "fps", "duration", "ping_pong", "global_tau"), _GRID)
def test_preview_export_tau_parity(
    mode: str, num_ts: int, loop_count: float, fps: float, duration: float, ping_pong: bool, global_tau: float
) -> None:
    """Preview and export produce the same τ sequence from the one schedule."""
    sched = TemporalSchedule(mode, fps, duration, num_ts, loop_count, ping_pong, global_tau)
    n = sched.n_frames(minimum=1)
    assert sched.max_preview_frame() == max(1, n - 1)

    def path_tau(i: int) -> float:
        return i / max(1, n - 1) - 0.5

    preview_seq = [sched.tau(i, path_tau(i)) for i in range(sched.max_preview_frame() + 1)]
    export_seq = [sched.tau(i, path_tau(i)) for i in range(n)]
    assert preview_seq == export_seq[: len(preview_seq)]


def test_freeze_constant_tau() -> None:
    """Freeze mode returns global_tau for every frame regardless of i/path_tau."""
    sched = TemporalSchedule("Freeze", 30.0, 4.0, 100, global_tau=0.3)
    assert sched.tau(0, -0.5) == pytest.approx(0.3)
    assert sched.tau(50, 0.5) == pytest.approx(0.3)


def test_ping_pong_loop_reverses_every_other_sweep() -> None:
    sched = TemporalSchedule("Loop", 30.0, 4.0, 5, loop_count=2.0, ping_pong=True)
    assert [sched.tau(i, 0.0) for i in (0, 4, 5, 9)] == pytest.approx([-0.5, 0.5, 0.5, -0.5])
