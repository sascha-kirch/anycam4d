"""DyNeRF importer: static multi-camera rig (per-camera ``camNN.mp4``) -> canonical ``opencv`` dataset.

DyNeRF (Neural 3D Video) is a synchronized static rig, so it is a :class:`VideoRigImporter` with DyNeRF's naming:
camera ids come from the ``camNN.mp4`` file names (sparse ids such as a missing ``cam03`` are kept), and COLMAP
calibrates frame 0 with the OPENCV model, one camera per video. This recovers Brown-Conrady distortion that the
released ``poses_bounds.npy`` (pinhole, LLFF convention) lacks, so that file is not read.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from anycam4d.data.prep.importers.video_rig import VideoRigImporter
from anycam4d.enums import CameraModelType


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from anycam4d.data.prep.colmap import ColmapOptions


_CAM_RE = re.compile(r"cam(\d+)\.mp4$")


class DyNeRFImporter(VideoRigImporter):
    """Turns a DyNeRF scene directory (``cam*.mp4`` files) into a canonical opencv dataset."""

    def __init__(
        self,
        source_dir: Path,
        work_dir: Path,
        *,
        fps_cap: float = 30.0,
        exclude_cams: Iterable[int] = (),
        max_frames: int | None = None,
        colmap: ColmapOptions | None = None,
    ) -> None:
        """Configure the importer (see :class:`VideoRigImporter` for the arguments)."""
        super().__init__(
            source_dir,
            work_dir,
            camera_model=CameraModelType.OPENCV,
            fps_cap=fps_cap,
            exclude_cams=exclude_cams,
            max_frames=max_frames,
            colmap=colmap,
        )

    def _find_videos(self) -> dict[int, Path]:
        """Camera id parsed from each ``camNN.mp4``.

        Raises:
            FileNotFoundError: If no ``camNN.mp4`` is present.
        """
        found = {}
        for path in sorted(self._source_dir.glob("cam*.mp4")):
            match = _CAM_RE.search(path.name)
            if match is not None:
                found[int(match.group(1))] = path
        if not found:
            raise FileNotFoundError(f"No camNN.mp4 videos found in {self._source_dir}")
        return found
