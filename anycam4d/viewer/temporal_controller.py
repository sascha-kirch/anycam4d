"""Single owner of live τ and the one playback timer.

:meth:`TemporalController.set_tau` is the only write path; it syncs the slider without re-entering its
callback and requests one render. One gated worker drives both τ playback and camera-path preview.
Pacing is elapsed-time based. ``fps`` is the live rate, separate from the export rate.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from typing import Callable


if TYPE_CHECKING:
    import viser


class TemporalController:
    """Owns live τ, fps, timesteps, play/loop/speed state and the playback timer.

    Args:
        fps: Live playback rate.
        num_timesteps: Discrete timesteps (frame↔τ, slider step).
        rerender: Callback requesting one render.
        initial_tau: Starting timestamp.
    """

    def __init__(
        self,
        *,
        fps: float,
        num_timesteps: int,
        rerender: Callable[[], None],
        initial_tau: float = -0.5,
    ) -> None:
        """Initialize the controller; widgets are attached later via :meth:`attach_widgets`."""
        self.fps = float(fps)
        self.num_timesteps = int(num_timesteps)
        self.loop = True
        self.speed = 1.0
        self.playing = False

        self._tau = float(initial_tau)
        self._rerender = rerender

        # Widgets are attached after the GUI is built; all optional for headless use.
        self._slider: viser.GuiInputHandle | None = None
        self._frame_display: viser.GuiInputHandle | None = None
        self._play_button: viser.GuiInputHandle | None = None

        # True only while we programmatically write the slider, so the slider's own callback
        # can short-circuit and not trigger a second render / re-enter this write.
        self._suppress_slider_cb = False

        # Single gated playback worker.
        self._job: str | None = None  # None | "live" | "preview"
        self._preview_tick: Callable[[], None] | None = None
        self._preview_period: Callable[[], float] | None = None
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def attach_widgets(
        self,
        *,
        slider: viser.GuiInputHandle,
        frame_display: viser.GuiInputHandle,
        play_button: viser.GuiInputHandle,
    ) -> None:
        """Bind the τ slider, Frame display, and Play button this controller drives."""
        self._slider = slider
        self._frame_display = frame_display
        self._play_button = play_button

    @property
    def tau(self) -> float:
        """The current live timestamp τ (read-only; write via :meth:`set_tau`)."""
        return self._tau

    @property
    def suppressing_slider_cb(self) -> bool:
        """True while a programmatic slider write is in flight (slider callback must no-op)."""
        return self._suppress_slider_cb

    def frame_from_tau(self, tau: float) -> int:
        """Map τ ∈ [-0.5, 0.5] to a frame index."""
        return round((tau + 0.5) * self.num_timesteps)

    def set_tau(self, tau: float, *, rerender: bool = True) -> None:
        """Set τ, update the frame display and slider without re-entry, and optionally request one render."""
        self._tau = float(tau)
        if self._frame_display is not None:
            self._frame_display.value = self.frame_from_tau(self._tau)
        if self._slider is not None:
            self._suppress_slider_cb = True
            try:
                self._slider.value = self._tau
            finally:
                self._suppress_slider_cb = False
        if rerender:
            self._rerender()

    @staticmethod
    def _playback_sleep_seconds(speed: float, fps: float) -> float:
        """Wall-clock seconds per playback tick; speed is applied only here (rate linear in speed)."""
        return 1.0 / (fps * speed)

    def _tau_per_second(self) -> float:
        """Live τ advance per real second: per-timestep increment over the per-tick sleep."""
        increment = 1.0 / self.num_timesteps
        return increment / self._playback_sleep_seconds(self.speed, self.fps)

    def toggle_live(self) -> None:
        """Toggle live-τ playback on/off (Play button)."""
        if self._job == "live":
            self.pause()
        else:
            self.play_live()

    def play_live(self) -> None:
        """Start live-τ playback (replaces any preview playback on the shared timer)."""
        with self._lock:
            self._job = "live"
            self.playing = True
            self._set_play_label(playing=True)
            self._ensure_thread()
            self._wake.set()

    def pause(self) -> None:
        """Stop live-τ playback."""
        with self._lock:
            self._job = None
            self.playing = False
            self._set_play_label(playing=False)

    def play_preview(self, tick: Callable[[], None], period: Callable[[], float]) -> None:
        """Drive camera-path preview on the shared timer: ``tick`` advances a frame, ``period`` gives the wait."""
        with self._lock:
            self.playing = False
            self._set_play_label(playing=False)
            self._preview_tick = tick
            self._preview_period = period
            self._job = "preview"
            self._ensure_thread()
            self._wake.set()

    def stop_preview(self) -> None:
        """Stop camera-path preview playback."""
        with self._lock:
            if self._job == "preview":
                self._job = None
            self._preview_tick = None
            self._preview_period = None

    def _set_play_label(self, *, playing: bool) -> None:
        """Reflect the live playing state on the Play button label."""
        if self._play_button is not None:
            self._play_button.label = "⏸ Pause" if playing else "▶ Play"

    def _ensure_thread(self) -> None:
        """Start the single playback worker on first use (caller holds ``_lock``)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="temporal-playback")
            self._thread.start()

    def _run(self) -> None:
        """The one playback loop: gated, single-threaded, dispatches by job mode."""
        last = time.monotonic()
        while True:
            job = self._job
            if job is None:
                self._wake.wait()
                self._wake.clear()
                last = time.monotonic()
                continue
            now = time.monotonic()
            dt = now - last
            last = now
            if job == "live":
                self._advance_live(dt)
                period = self._playback_sleep_seconds(self.speed, self.fps)
            else:
                tick = self._preview_tick
                if tick is not None:
                    tick()
                period_fn = self._preview_period
                period = period_fn() if period_fn is not None else 1.0 / max(self.fps, 1.0)
            time.sleep(max(0.0, period))

    def _advance_live(self, dt: float) -> None:
        """Advance live τ by the elapsed-time-paced increment; wrap on loop or stop at the end."""
        new_tau = self._tau + self._tau_per_second() * dt
        if new_tau > 0.5:
            if self.loop:
                new_tau = -0.5
            else:
                self.set_tau(0.5)
                self.pause()
                return
        self.set_tau(new_tau)
