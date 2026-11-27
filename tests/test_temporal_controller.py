"""``TemporalController`` with fake widgets: one tau owner, one gated playback thread."""

import threading
import time

from anycam4d.viewer.temporal_controller import TemporalController


class _FakeWidget:
    """Minimal stand-in for a viser GUI handle with a settable ``value`` / ``label``."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.label = ""


class _CallbackSlider:
    """Fake slider that fires its callback on every programmatic write, like viser."""

    def __init__(self) -> None:
        self._value = 0.0
        self.on_change = None

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float) -> None:
        self._value = v
        if self.on_change is not None:
            self.on_change()


def _make_controller(rerender=None, *, num_timesteps: int = 100) -> TemporalController:
    return TemporalController(
        fps=30.0,
        num_timesteps=num_timesteps,
        rerender=rerender or (lambda: None),
    )


def test_frame_from_tau_boundaries() -> None:
    """τ endpoints map to frame 0 and num_timesteps."""
    c = _make_controller()
    assert c.frame_from_tau(-0.5) == 0
    assert c.frame_from_tau(0.5) == 100
    assert c.frame_from_tau(0.0) == 50


def test_set_tau_updates_frame_and_requests_one_render() -> None:
    """set_tau updates the Frame display and triggers exactly one render."""
    calls = []
    c = _make_controller(rerender=lambda: calls.append(1))
    frame_display = _FakeWidget()
    c.attach_widgets(slider=_FakeWidget(), frame_display=frame_display, play_button=_FakeWidget())

    c.set_tau(0.0)

    assert c.tau == 0.0
    assert frame_display.value == 50
    assert len(calls) == 1


def test_slider_back_sync_does_not_reenter_callback() -> None:
    """A programmatic slider write is suppressed: the slider callback does not re-drive set_tau."""
    set_tau_calls = []
    c = _make_controller()
    slider = _CallbackSlider()

    # Viewer-style slider callback: ignore the echo from our own write.
    def _slider_cb() -> None:
        if c.suppressing_slider_cb:
            return
        set_tau_calls.append(slider.value)
        c.set_tau(slider.value)

    slider.on_change = _slider_cb
    c.attach_widgets(slider=slider, frame_display=_FakeWidget(), play_button=_FakeWidget())

    c.set_tau(0.25)  # programmatic write → callback must short-circuit

    assert slider.value == 0.25
    assert set_tau_calls == []  # no re-entry


def test_playback_speed_applied_once_and_monotonic() -> None:
    """τ advance per real second is linear in speed (applied exactly once)."""
    n = 100
    increment = 1.0 / n

    def rate(speed: float) -> float:
        return increment / TemporalController._playback_sleep_seconds(speed, 30.0)

    assert rate(2.0) > rate(1.0)
    assert rate(2.0) == 2.0 * rate(1.0)


def _live_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "temporal-playback")


def test_single_playback_thread_no_leak() -> None:
    """Rapid play/pause spawns at most one worker (measured against the thread baseline)."""
    baseline = _live_threads()
    c = _make_controller()
    c.attach_widgets(slider=_FakeWidget(), frame_display=_FakeWidget(), play_button=_FakeWidget())

    for _ in range(20):
        c.toggle_live()
        c.toggle_live()

    assert _live_threads() - baseline <= 1

    c.pause()
    time.sleep(0.05)
    assert _live_threads() - baseline <= 1


def test_pause_gate_stops_playback() -> None:
    """After pause, the gate idles: playing is False and this controller adds ≤1 worker."""
    baseline = _live_threads()
    c = _make_controller()
    c.attach_widgets(slider=_FakeWidget(), frame_display=_FakeWidget(), play_button=_FakeWidget())
    c.play_live()
    assert c.playing is True
    c.pause()
    assert c.playing is False
    assert _live_threads() - baseline <= 1
