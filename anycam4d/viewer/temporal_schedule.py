"""TemporalSchedule: the one mapping from a temporal mode to frame counts and the per-frame tau.

``base_n_frames`` is one tau sweep (``num_timesteps`` in Loop mode, else ``fps * duration``) and is the period of
:meth:`tau`; ``n_frames`` is the length of the video or preview (``loop_count * num_timesteps`` in Loop mode). ``fps``
is the export frame rate, seeded from but independent of the live playback rate of :class:`TemporalController`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalSchedule:
    """Pure mapping from a temporal mode + timing to frame counts and per-frame τ.

    Args:
        mode: One of ``"Manual"``, ``"Loop"``, ``"Freeze"``.
        fps: Export frames-per-second (frames = ``fps * duration`` outside Loop mode).
        duration: Camera-path duration in seconds.
        num_timesteps: Number of discrete timesteps in the sequence (one τ sweep in Loop mode).
        loop_count: Number of full τ sweeps over the video (Loop mode); may be fractional.
        ping_pong: Reverse sweep direction on odd loops instead of jumping back to start (Loop mode).
        global_tau: Constant τ applied to every frame (Freeze mode).
    """

    mode: str
    fps: float
    duration: float
    num_timesteps: int
    loop_count: float = 1.0
    ping_pong: bool = False
    global_tau: float = -0.5

    def base_n_frames(self) -> int:
        """Frames in one τ sweep / camera orbit (the period used by :meth:`tau`)."""
        if self.mode == "Loop":
            return max(1, self.num_timesteps)
        return max(1, int(self.fps * self.duration))

    def n_frames(self, minimum: int = 1) -> int:
        """Total frames to render (Loop: ``loop_count * num_timesteps``; else ``fps * duration``).

        Args:
            minimum: Floor for the returned count (display/export use 1, spline viz uses 2).
        """
        if self.mode == "Loop":
            return max(minimum, round(self.loop_count * self.num_timesteps))
        return max(minimum, int(self.fps * self.duration))

    def max_preview_frame(self) -> int:
        """Maximum preview-slider frame index (``n_frames - 1``, at least 1)."""
        if self.mode == "Loop" and self.num_timesteps > 0:
            return max(1, round(self.loop_count * self.num_timesteps) - 1)
        return max(1, int(self.fps * self.duration) - 1)

    def tau(self, i: int, path_tau: float) -> float:
        """τ for frame ``i``.

        Args:
            i: Frame index (0-based).
            path_tau: τ interpolated from the camera path (used only in Manual mode).

        Returns:
            τ in [-0.5, 0.5].
        """
        if self.mode == "Freeze":
            return self.global_tau
        if self.mode == "Loop":
            base = self.base_n_frames()
            loop_i = i % base
            loop_idx = i // base
            phase = loop_i / max(1, base - 1)
            if self.ping_pong and loop_idx % 2 == 1:
                phase = 1.0 - phase
            return phase - 0.5
        return path_tau
