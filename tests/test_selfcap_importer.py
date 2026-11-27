"""``SelfCapImporter`` on synthetic fixtures; the real release import is ``slow``."""

from __future__ import annotations

import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import cv2

from anycam4d.data.prep.importers.selfcap import SelfCapImporter
from anycam4d.data.prep.build_dataset import build_dataset
from anycam4d.enums import CameraModelType
from anycam4d.point_ops.ply import write_ply


# Solid RGB per frame index so BGR<->RGB order + frame ordering survive the lossy mp4 -> jpg round-trip.
_FRAME_COLORS_RGB = [(230, 10, 10), (10, 230, 10), (10, 10, 230), (230, 230, 10), (10, 230, 230)]

_K = np.array([[500.0, 0.0, 16.0], [0.0, 500.0, 12.0], [0.0, 0.0, 1.0]])


def _write_video(path: Path, n_frames: int, size: tuple[int, int] = (32, 24), fps: float = 60.0) -> None:
    """Write a solid-color-per-frame mp4 (frame f uses ``_FRAME_COLORS_RGB[f]``)."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(n_frames):
        r, g, b = _FRAME_COLORS_RGB[f]
        bgr = np.empty((h, w, 3), dtype=np.uint8)
        bgr[..., 0], bgr[..., 1], bgr[..., 2] = b, g, r
        writer.write(bgr)
    writer.release()


def _write_calib(source: Path, cams: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """Write ``optimized/intri.yml`` + ``extri.yml`` for ``cams`` = [(name, K, dist5, Rot, T), ...]."""
    opt = source / "optimized"
    opt.mkdir(parents=True, exist_ok=True)
    intri = cv2.FileStorage(str(opt / "intri.yml"), cv2.FILE_STORAGE_WRITE)
    extri = cv2.FileStorage(str(opt / "extri.yml"), cv2.FILE_STORAGE_WRITE)
    for name, k, dist, rot, t in cams:
        intri.write(f"K_{name}", k)
        intri.write(f"D_{name}", dist.reshape(1, 5))
        extri.write(f"Rot_{name}", rot)
        extri.write(f"T_{name}", t.reshape(3, 1))
    intri.release()
    extri.release()


def _make_scene(
    source: Path,
    names: list[str],
    n_frames: int,
    *,
    dist: np.ndarray | None = None,
    dense_frames: tuple[int, ...] = (0,),
    fps: float = 60.0,
) -> None:
    """Build a full synthetic SelfCap directory: calib + per-camera videos + dense_pcds."""
    dist = np.zeros(5) if dist is None else dist
    cams = []
    (source / "videos").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        rot = Rotation.from_euler("xyz", [0.1 * i, -0.2 * i, 0.05 * i]).as_matrix()
        t = np.array([0.5 * i, 0.1, 3.0])
        cams.append((name, _K.copy(), dist.copy(), rot, t))
        _write_video(source / "videos" / f"{name}.mp4", n_frames, fps=fps)
    _write_calib(source, cams)

    # Points are filled with their own frame number so a test can read back which cloud was chosen.
    if dense_frames:
        pcd_dir = source / "dense_pcds"
        pcd_dir.mkdir(exist_ok=True)
        for fnum in dense_frames:
            xyz = np.full((4, 3), float(fnum), dtype=np.float32)
            rgb = np.full((4, 3), 100, dtype=np.uint8)
            write_ply(pcd_dir / f"{fnum:06d}.ply", xyz, rgb)


def _dominant_channel(image_path: Path) -> int:
    """Dominant RGB channel of a written image (proves BGR<->RGB order + frame identity)."""
    bgr = cv2.imread(str(image_path))
    return int(np.argmax(bgr[..., ::-1].reshape(-1, 3).mean(axis=0)))


def test_calibration_parse_and_w2c_to_c2w(tmp_path: Path) -> None:
    """Parsed c2w satisfies R_c2w=Rot^T, C=-Rot^T@T; c2w inverts world->camera (no axis flip)."""
    _make_scene(tmp_path, ["02", "05"], n_frames=2)
    importer = SelfCapImporter(tmp_path)
    importer.prepare()

    for cam_id in (2, 5):
        calib = importer._calib[cam_id]  # noqa: SLF001
        i = {2: 0, 5: 1}[cam_id]
        rot = Rotation.from_euler("xyz", [0.1 * i, -0.2 * i, 0.05 * i]).as_matrix()
        t = np.array([0.5 * i, 0.1, 3.0])
        np.testing.assert_allclose(calib.c2w[:3, :3], rot.T, atol=1e-5)
        np.testing.assert_allclose(calib.c2w[:3, 3], -rot.T @ t, atol=1e-5)
        x_world = np.array([0.3, -0.4, 1.2])
        x_cam = rot @ x_world + t
        back = calib.c2w[:3, :3] @ x_cam + calib.c2w[:3, 3]
        np.testing.assert_allclose(back, x_world, atol=1e-5)
        assert calib.params[0] == pytest.approx(500.0)  # fx
        assert calib.params[2] == pytest.approx(16.0)  # cx


def test_model_always_opencv_with_zero_distortion(tmp_path: Path) -> None:
    """Zero distortion still emits model 'opencv' with 9 params (not collapsed to pinhole); loads."""
    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset

    _make_scene(tmp_path, ["00", "01"], n_frames=2)
    importer = SelfCapImporter(tmp_path, fps_cap=60.0)
    for cam in importer.cameras():
        assert cam.model is CameraModelType.OPENCV
        assert len(cam.params) == 9
        assert cam.params[4:] == [0.0, 0.0, 0.0, 0.0, 0.0]

    out = tmp_path / "out"
    build_dataset(importer, out, image_format="jpg")
    ds = MultiViewSequenceDataset(out, split="train")
    assert ds.camera_model is CameraModelType.OPENCV
    assert ds.num_timesteps == 2


def test_noncontiguous_camera_ids(tmp_path: Path) -> None:
    """yml keys K_02/K_05/K_10 -> cam ids {2,5,10}; unsorted node order handled."""
    _make_scene(tmp_path, ["10", "02", "05"], n_frames=2)  # deliberately unsorted
    importer = SelfCapImporter(tmp_path)
    importer.prepare()
    assert importer._cam_ids == [2, 5, 10]  # noqa: SLF001


def test_trim_and_extraction(tmp_path: Path) -> None:
    """start_frame=1,end_frame=3 on 5-frame 60fps videos, no decimation -> frames 1,2; BGR->RGB correct."""
    _make_scene(tmp_path, ["00", "01"], n_frames=5)
    importer = SelfCapImporter(tmp_path, fps_cap=60.0, start_frame=1, end_frame=3)
    out = tmp_path / "out"
    build_dataset(importer, out, image_format="jpg")

    assert importer._count == 2  # noqa: SLF001
    for cid in (0, 1):
        # Output timestep 0 == source frame 1, timestep 1 == source frame 2 (dominant channel proves it).
        assert _dominant_channel(out / f"images/cam{cid:02d}_0000.jpg") == 1
        assert _dominant_channel(out / f"images/cam{cid:02d}_0001.jpg") == 2
    doc = json.loads((out / "train_dataset.json").read_text())
    timestamps = sorted({c["timestamp"] for c in doc["captures"]})
    assert timestamps == pytest.approx([0.0, 1.0 / 60.0])  # rebased to output index / effective fps


def test_decimation_over_span(tmp_path: Path) -> None:
    """60fps capped at 30 keeps every 2nd frame; over [0,3) that is source frames 0, 2 (R, B)."""
    _make_scene(tmp_path, ["00"], n_frames=3, fps=60.0)  # frames 0,1,2 == clean R,G,B
    importer = SelfCapImporter(tmp_path, fps_cap=30.0)
    out = tmp_path / "out"
    build_dataset(importer, out, image_format="jpg")

    assert importer._decim_k == 2  # noqa: SLF001
    assert importer._count == 2  # noqa: SLF001  ceil(3 / 2)
    assert _dominant_channel(out / "images/cam00_0000.jpg") == 0  # source frame 0 -> red
    assert _dominant_channel(out / "images/cam00_0001.jpg") == 2  # source frame 2 -> blue


def test_max_frames_clamps(tmp_path: Path) -> None:
    """max_frames caps the output frame count below the trim window."""
    _make_scene(tmp_path, ["00", "01"], n_frames=5)
    importer = SelfCapImporter(tmp_path, fps_cap=60.0, start_frame=1, max_frames=1)
    importer.prepare()
    assert importer._count == 1  # noqa: SLF001
    assert len(importer.captures()) == 1 * 2  # 1 frame x 2 cameras


def test_dense_pcd_nearest_start(tmp_path: Path) -> None:
    """The init cloud is the dense cloud nearest start_frame (start=900 -> 001000)."""
    _make_scene(tmp_path, ["00", "01"], n_frames=2, dense_frames=(0, 1000))
    importer = SelfCapImporter(tmp_path, start_frame=900)
    xyz, rgb = importer.point_cloud()
    np.testing.assert_allclose(xyz, 1000.0)  # points were filled with their frame number


def test_point_cloud_none_without_any_cloud(tmp_path: Path) -> None:
    _make_scene(tmp_path, ["00", "01"], n_frames=2, dense_frames=())
    assert SelfCapImporter(tmp_path).point_cloud() is None


def test_timestamps_are_shared_across_cameras(tmp_path: Path) -> None:
    """``sync.json`` sub-frame offsets never reach the timestamps, which key frames."""
    _make_scene(tmp_path, ["00", "01", "02"], n_frames=3)
    with open(tmp_path / "optimized" / "sync.json", "w") as f:
        json.dump({"00": 0.5, "01": -0.25}, f)  # would desync every camera if applied

    importer = SelfCapImporter(tmp_path, fps_cap=60.0)
    caps = importer.captures()

    by_frame = defaultdict(set)
    for cap in caps:
        by_frame[cap.frame_id].add(cap.timestamp)
    assert set(by_frame) == {0, 1, 2}
    assert all(len(stamps) == 1 for stamps in by_frame.values())  # one timestamp per timestep
    assert sorted(next(iter(s)) for s in by_frame.values()) == pytest.approx([0.0, 1 / 60.0, 2 / 60.0])


@pytest.fixture(scope="session")
def selfcap_bar_dir() -> Path:
    """Path to the real SelfCap bar-release (``ANYCAM4D_SELFCAP_DIR``), or skip if absent."""
    path = Path(os.environ.get("ANYCAM4D_SELFCAP_DIR", "data/raw/selfcap/bar-release"))
    if not path.exists() or not (path / "optimized" / "intri.yml").exists():
        pytest.skip("SelfCap bar-release not available")
    return path


@pytest.mark.slow
def test_end_to_end_import_builds_loadable_dataset(selfcap_bar_dir: Path, tmp_path: Path) -> None:
    """Real release -> canonical opencv dataset that MultiViewSequenceDataset loads (gated)."""
    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset

    importer = SelfCapImporter(selfcap_bar_dir, max_frames=2)
    out = tmp_path / "out"
    build_dataset(importer, out, eval_cams=(7,), image_format="jpg")

    ds = MultiViewSequenceDataset(out, split="train")
    assert ds.num_timesteps == 2
    assert ds.camera_model is CameraModelType.OPENCV
