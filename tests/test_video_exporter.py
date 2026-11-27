"""``VideoExporter`` with viser, ffmpeg and rendering stubbed, run synchronously."""

from __future__ import annotations

import contextlib
import types

import numpy as np

import anycam4d.viewer.video_exporter as ve_module
from anycam4d.viewer.temporal_schedule import TemporalSchedule
from anycam4d.viewer.video_exporter import VideoExporter


class _StubFolder:
    """Context-manager stand-in for a viser folder."""

    def __enter__(self) -> _StubFolder:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _StubProgress:
    """Progress-bar stand-in recording its final value and removal."""

    def __init__(self) -> None:
        self.value = 0.0
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _StubServer:
    """Server stand-in exposing only gui.add_progress_bar."""

    def __init__(self, progress: _StubProgress) -> None:
        self.gui = types.SimpleNamespace(add_progress_bar=lambda *a, **k: progress)


class _StubViewer:
    """Viewer stand-in capturing the τ sequence rendered under its lock."""

    def __init__(self) -> None:
        self.taus: list[float] = []
        self.lock = contextlib.nullcontext()

    def render_frame(self, *, c2w, fov_rad, width, height, timestamp):  # noqa: ANN001, ARG002
        self.taus.append(timestamp)
        return np.zeros((height, width, 3), dtype=np.uint8)


class _StubProc:
    """subprocess.Popen stand-in capturing the number of frames written."""

    def __init__(self) -> None:
        self.writes = 0
        self.stdin = types.SimpleNamespace(write=self._write, close=lambda: None)

    def _write(self, _data: bytes) -> None:
        self.writes += 1

    def wait(self) -> int:
        return 0


class _StubCameraPath:
    """Camera path returning a frame whose timestamp encodes the frame index."""

    def interpolate(self, normalized_t: float):  # noqa: ANN201
        pose = types.SimpleNamespace(as_matrix=lambda: np.eye(4))
        return types.SimpleNamespace(pose=pose, fov_rad=1.0, timestamp=normalized_t)


def _run_synchronously(monkeypatch) -> None:  # noqa: ANN001
    """Patch the exporter's Thread so .start() runs the target inline."""

    class _SyncThread:
        def __init__(self, target, daemon=False) -> None:  # noqa: ANN001, ARG002
            self._target = target

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(ve_module.threading, "Thread", _SyncThread)


def test_export_drives_one_render_per_frame(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Freeze mode renders exactly n_frames frames at the constant global τ and writes each."""
    proc = _StubProc()
    monkeypatch.setattr(ve_module.subprocess, "Popen", lambda *a, **k: proc)
    _run_synchronously(monkeypatch)

    progress = _StubProgress()
    viewer = _StubViewer()
    exporter = VideoExporter(_StubServer(progress), _StubFolder(), viewer)
    schedule = TemporalSchedule(mode="Freeze", fps=10, duration=2.0, num_timesteps=100, global_tau=0.25)
    n_frames = schedule.n_frames(minimum=1)

    exporter.export(_StubCameraPath(), schedule, 8, 6, 10, tmp_path / "out.mp4", controls_to_disable=[])

    assert len(viewer.taus) == n_frames
    assert proc.writes == n_frames
    assert all(t == 0.25 for t in viewer.taus)  # Freeze τ is constant
    assert progress.removed


def test_export_restores_disabled_handles(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Controls are disabled during export and restored to their prior state afterward."""
    monkeypatch.setattr(ve_module.subprocess, "Popen", lambda *a, **k: _StubProc())
    _run_synchronously(monkeypatch)

    handle_enabled = types.SimpleNamespace(disabled=False)
    handle_already = types.SimpleNamespace(disabled=True)
    exporter = VideoExporter(_StubServer(_StubProgress()), _StubFolder(), _StubViewer())
    schedule = TemporalSchedule(mode="Freeze", fps=5, duration=1.0, num_timesteps=100)

    exporter.export(
        _StubCameraPath(), schedule, 8, 6, 5, tmp_path / "o.mp4", controls_to_disable=[handle_enabled, handle_already]
    )

    assert handle_enabled.disabled is False  # restored to prior enabled state
    assert handle_already.disabled is True  # restored to prior disabled state
