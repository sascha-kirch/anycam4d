"""Geometry-only 4D camera path: keyframes, splines, interpolation, serialization. No viser calls; ``camera_path_view`` draws."""

from __future__ import annotations

import dataclasses
import threading
from typing import TYPE_CHECKING

import numpy as np
import splines
import splines.quaternion
import viser.transforms as tf
from scipy import interpolate
from scipy.spatial.transform import Rotation

from anycam4d.viewer.constants import DEFAULT_ASPECT


if TYPE_CHECKING:
    from collections.abc import Callable

    import viser
    from jaxtyping import Float


MIN_TRANSITION_SEC = 0.001


@dataclasses.dataclass
class Keyframe4D:
    """One keyframe: scene-space pose (wxyz), aspect, tau in ``[-0.5, 0.5]`` and an optional transition override in seconds."""

    position: Float[np.ndarray, "3"]
    wxyz: Float[np.ndarray, "4"]
    aspect: float
    timestamp: float
    override_transition_enabled: bool
    override_transition_sec: float | None

    @staticmethod
    def from_camera(camera: viser.CameraHandle, aspect: float, timestamp: float) -> Keyframe4D:
        """Keyframe from the current viser camera, without a transition override."""
        return Keyframe4D(
            position=camera.position,
            wxyz=camera.wxyz,
            aspect=aspect,
            timestamp=timestamp,
            override_transition_enabled=False,
            override_transition_sec=None,
        )


@dataclasses.dataclass(frozen=True)
class InterpolatedFrame:
    """Camera-to-world pose, vertical FOV in radians and tau at one point on the path."""

    pose: tf.SE3
    fov_rad: float
    timestamp: float


class CameraPathModel:
    """Keyframes interpolated by Kochanek-Bartels splines (position, tau) and SLERP (rotation).

    High-frequency edits call :meth:`mark_spline_dirty`, which coalesces rebuilds behind one trailing timer;
    reads flush synchronously, and :attr:`on_rebuilt` fires after each rebuild.
    """

    def __init__(self) -> None:
        """Initialize an empty camera path with no keyframes."""
        self._keyframes: dict[int, Keyframe4D] = {}
        self._keyframe_counter: int = 0

        self._position_spline: splines.KochanekBartels | None = None
        self._orientation_spline: splines.quaternion.KochanekBartels | None = None
        self._timestamp_spline: splines.KochanekBartels | None = None
        self._time_to_spline: interpolate.PchipInterpolator | None = None

        # Coalesced rebuilds: 50 ms swallows a drag burst without perceptible lag.
        self._rebuild_quiet_sec: float = 0.05
        self._spline_dirty: bool = False
        self._rebuild_timer: threading.Timer | None = None
        self._rebuild_lock = threading.Lock()

        # Invoked (under no lock) after every interpolator rebuild so a view can redraw the
        # spline visualization. None when running headless / without a view.
        self.on_rebuilt: Callable[[], None] | None = None

        self.default_transition_sec: float = 2.0
        self.default_fov_rad: float = 50.0 * np.pi / 180.0
        self.default_timestamp: float = -0.5
        self.tension: float = 0.5
        self.tau_editable: bool = True
        self.tau_value_fn: Callable[[int], float] | None = None

    def add_keyframe(self, keyframe: Keyframe4D, keyframe_index: int | None = None) -> int:
        """Store a keyframe at ``keyframe_index`` or the next free index; returns the index."""
        if keyframe_index is None:
            keyframe_index = self._keyframe_counter
            self._keyframe_counter += 1
        self._keyframes[keyframe_index] = keyframe
        return keyframe_index

    def remove_keyframe(self, keyframe_index: int) -> None:
        """Remove a keyframe by index (no-op if absent)."""
        self._keyframes.pop(keyframe_index, None)

    def get(self, keyframe_index: int) -> Keyframe4D | None:
        """Return the keyframe at ``keyframe_index`` or None."""
        return self._keyframes.get(keyframe_index)

    def keyframes(self) -> list[tuple[int, Keyframe4D]]:
        """Return ``(index, keyframe)`` pairs in insertion order."""
        return list(self._keyframes.items())

    def sorted_keyframe_indices(self) -> list[int]:
        """Return keyframe indices in ascending order."""
        return sorted(self._keyframes.keys())

    def clear(self) -> None:
        """Remove all keyframes (data only)."""
        self._keyframes.clear()

    def _get_transition_sec(self, kf: Keyframe4D) -> float:
        """Effective transition time of ``kf`` in seconds."""
        if kf.override_transition_enabled and kf.override_transition_sec is not None:
            return max(kf.override_transition_sec, MIN_TRANSITION_SEC)
        return max(self.default_transition_sec, MIN_TRANSITION_SEC)

    def compute_duration(self) -> float:
        """Total path duration in seconds; 0 with fewer than two keyframes."""
        keyframes = list(self._keyframes.values())
        if len(keyframes) < 2:
            return 0.0
        total = 0.0
        for i, kf in enumerate(keyframes):
            if i > 0:
                total += self._get_transition_sec(kf)
        return total

    def _build_time_mapping(self) -> interpolate.PchipInterpolator | None:
        """PCHIP map from elapsed seconds to spline parameter, or ``None`` with fewer than two keyframes."""
        keyframes = list(self._keyframes.values())
        if len(keyframes) < 2:
            return None

        time_pts: list[float] = [0.0]
        spline_pts: list[float] = [0.0]
        t = 0.0

        for i in range(1, len(keyframes)):
            kf = keyframes[i]
            t += self._get_transition_sec(kf)
            time_pts.append(t)
            spline_pts.append(float(i))

        return interpolate.PchipInterpolator(time_pts, spline_pts)

    def _rebuild_splines(self) -> bool:
        """Recompute the splines; returns whether at least two keyframes allowed it."""
        keyframes = list(self._keyframes.values())
        if len(keyframes) < 2:
            self._position_spline = None
            self._orientation_spline = None
            self._timestamp_spline = None
            self._time_to_spline = None
            return False

        tcb = (self.tension, 0.0, 0.0)

        self._position_spline = splines.KochanekBartels(
            [kf.position for kf in keyframes],
            tcb=tcb,
            endconditions="natural",
        )
        self._orientation_spline = splines.quaternion.KochanekBartels(
            [splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(kf.wxyz, -1)) for kf in keyframes],
            tcb=tcb,
            endconditions="natural",
        )
        timestamps = [kf.timestamp for kf in keyframes]
        self._timestamp_spline = splines.KochanekBartels(timestamps, tcb=tcb, endconditions="natural")
        self._time_to_spline = self._build_time_mapping()
        return True

    def sample_path_positions(self, num_frames: int) -> Float[np.ndarray, "num_frames 3"] | None:
        """``[num_frames, 3]`` positions evenly spaced in time, or ``None`` without a spline."""
        if self._position_spline is None or self._time_to_spline is None:
            return None
        duration = self.compute_duration()
        t_sec = np.linspace(0.0, duration, num_frames)
        # Clamp to the spline's own grid: a refresh can fire before a new keyframe is rebuilt in.
        max_param = self._position_spline.grid[-1]
        spline_params = np.clip(self._time_to_spline(t_sec), 0, max_param)
        return self._position_spline.evaluate(spline_params)

    def update_spline(self) -> None:
        """Rebuild now and notify the view; for discrete actions, not drags."""
        self._rebuild_now(force=True)

    def mark_spline_dirty(self) -> None:
        """Mark the splines stale and re-arm the single trailing rebuild."""
        with self._rebuild_lock:
            self._spline_dirty = True
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()
            self._rebuild_timer = threading.Timer(self._rebuild_quiet_sec, self._on_rebuild_timer)
            self._rebuild_timer.start()

    def _on_rebuild_timer(self) -> None:
        """Trailing-timer callback: rebuild only if still dirty (settle flush)."""
        self._rebuild_now(force=False)

    def flush_pending_rebuild(self) -> None:
        """Run a pending deferred rebuild now, so reads never see stale splines."""
        self._rebuild_now(force=False)

    def _rebuild_now(self, *, force: bool) -> None:
        """Rebuild under the lock (unconditionally if ``force``, else only if still dirty), then notify outside it."""
        with self._rebuild_lock:
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()
                self._rebuild_timer = None
            if not force and not self._spline_dirty:
                return
            self._spline_dirty = False
            self._rebuild_splines()
        if self.on_rebuilt is not None:
            self.on_rebuilt()

    def interpolate(self, normalized_t: float) -> InterpolatedFrame | None:
        """Frame at ``normalized_t`` in ``[0, 1]``, or ``None`` with fewer than two keyframes."""
        self.flush_pending_rebuild()
        if self._position_spline is None or self._time_to_spline is None:
            return None

        duration = self.compute_duration()
        t_sec = normalized_t * duration
        spline_t = float(np.clip(self._time_to_spline(np.array(t_sec)), 0, len(self._keyframes) - 1))

        pos = self._position_spline.evaluate(spline_t)
        quat = self._orientation_spline.evaluate(spline_t)
        tau = float(self._timestamp_spline.evaluate(spline_t))

        pose = tf.SE3.from_rotation_and_translation(
            tf.SO3(np.array([quat.scalar, *quat.vector])),
            pos,
        )
        return InterpolatedFrame(pose=pose, fov_rad=self.default_fov_rad, timestamp=tau)

    def apply_lookat(self, target: Float[np.ndarray, "3"]) -> list[int]:
        """Turn every keyframe towards ``target`` in place; returns the indices that changed."""
        changed: list[int] = []
        for idx, kf in list(self._keyframes.items()):
            forward = target - kf.position
            dist = np.linalg.norm(forward)
            if dist < 1e-6:
                continue
            forward = forward / dist
            world_up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(forward, world_up)) > 0.999:
                world_up = np.array([0.0, 0.0, 1.0])
            z_cam = forward
            x_cam = np.cross(forward, world_up)
            x_cam = x_cam / np.linalg.norm(x_cam)
            y_cam = np.cross(z_cam, x_cam)
            y_cam = y_cam / np.linalg.norm(y_cam)
            R = np.stack([x_cam, y_cam, z_cam], axis=1)
            xyzw = Rotation.from_matrix(R).as_quat()
            kf.wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            changed.append(idx)
        return changed

    def to_json(self) -> list[dict]:
        """Keyframes as JSON-compatible dicts."""
        return [
            {
                "position": kf.position.tolist(),
                "wxyz": kf.wxyz.tolist(),
                "aspect": kf.aspect,
                "timestamp": kf.timestamp,
                "override_transition_enabled": kf.override_transition_enabled,
                "override_transition_sec": kf.override_transition_sec,
            }
            for kf in self._keyframes.values()
        ]

    def load_keyframes(self, keyframe_dicts: list[dict]) -> None:
        """Replace the keyframes with dicts from :meth:`to_json` (no drawing)."""
        self.clear()
        for kf_dict in keyframe_dicts:
            self.add_keyframe(
                Keyframe4D(
                    position=np.array(kf_dict["position"]),
                    wxyz=np.array(kf_dict["wxyz"]),
                    aspect=kf_dict.get("aspect", DEFAULT_ASPECT),
                    timestamp=kf_dict.get("timestamp", -0.5),
                    override_transition_enabled=kf_dict.get("override_transition_enabled", False),
                    override_transition_sec=kf_dict.get("override_transition_sec"),
                )
            )
