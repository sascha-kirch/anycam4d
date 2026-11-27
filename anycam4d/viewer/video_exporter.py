"""Offline camera-path video export: schedule iteration + ffmpeg piping.

Extracted from the render panel's inline ``_render_video`` closure. Takes explicit typed
arguments (camera path, schedule, resolution, output path, controls to disable) instead of a
long positional list and the old untyped handles dict.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
from typing import TYPE_CHECKING

from anycam4d.console import notice
from anycam4d.console import warn


if TYPE_CHECKING:
    from pathlib import Path

    import viser

    from anycam4d.viewer.camera_path_view import CameraPathView
    from anycam4d.viewer.temporal_schedule import TemporalSchedule
    from anycam4d.viewer.viewer import Anycam4DViewer


class VideoExporter:
    """Renders a camera path to an MP4 by piping raw frames to ffmpeg on a background thread.

    Args:
        server: Viser server (used to attach the progress bar).
        folder: Rendering folder the progress bar is created in.
        viewer: The 4DGS viewer providing ``lock`` and ``render_frame``.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        folder: viser.GuiFolderHandle,
        viewer: Anycam4DViewer,
    ) -> None:
        """Bind the exporter to a server/folder/viewer."""
        self._server = server
        self._folder = folder
        self._viewer = viewer

    def export(
        self,
        camera_path: CameraPathView,
        schedule: TemporalSchedule,
        width: int,
        height: int,
        fps: int,
        out_path: Path,
        controls_to_disable: list[viser.GuiInputHandle],
    ) -> None:
        """Render the path to ``out_path`` on a daemon thread.

        Disables ``controls_to_disable`` for the duration, shows a progress bar, iterates the
        schedule's frames (one ``render_frame`` per frame under ``viewer.lock``), pipes raw RGB
        to ffmpeg, then restores the controls' original disabled state.

        Args:
            camera_path: Camera path to interpolate.
            schedule: Temporal schedule giving frame count and per-frame τ.
            width: Output width in pixels.
            height: Output height in pixels.
            fps: Output frames per second.
            out_path: Destination MP4 path (parent created if needed).
            controls_to_disable: Handles disabled during render and restored afterward.
        """
        n_frames = schedule.n_frames(minimum=1)

        original_disabled = [getattr(h, "disabled", False) for h in controls_to_disable]
        for h in controls_to_disable:
            with contextlib.suppress(AttributeError):
                h.disabled = True

        with self._folder:
            progress = self._server.gui.add_progress_bar(0.0, animated=True)

        def _do_render() -> None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(  # noqa: S603
                self._ffmpeg_cmd(width, height, fps, out_path),
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            cam_denom = max(1, n_frames - 1)
            for i in range(n_frames):
                frame = camera_path.interpolate(i / cam_denom)
                if frame is None:
                    continue
                tau = schedule.tau(i, frame.timestamp)
                with self._viewer.lock:
                    rgb = self._viewer.render_frame(
                        c2w=frame.pose.as_matrix(),
                        fov_rad=frame.fov_rad,
                        width=width,
                        height=height,
                        timestamp=tau,
                    )
                proc.stdin.write(rgb.tobytes())
                progress.value = (i + 1) / n_frames * 100

            proc.stdin.close()
            retcode = proc.wait()
            if retcode != 0:
                warn(f"ffmpeg exited with code {retcode}")
            else:
                notice(f"Video saved: {out_path}")

            progress.remove()
            for h, orig in zip(controls_to_disable, original_disabled):
                with contextlib.suppress(AttributeError):
                    h.disabled = orig

        threading.Thread(target=_do_render, daemon=True).start()

    @staticmethod
    def _ffmpeg_cmd(width: int, height: int, fps: int, out_path: Path) -> list[str]:
        """Build the ffmpeg command that encodes piped raw RGB frames to H.264 MP4."""
        ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        return [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "fast",
            str(out_path),
        ]
