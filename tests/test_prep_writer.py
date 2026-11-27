"""Tests for anycam4d/data/prep/writer.py: PrepWriter round-trip, splits, provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.prep.sample import CalibratedView
from anycam4d.data.prep.writer import PrepWriter
from anycam4d.data.prep.provenance import build_manifest
from anycam4d.data.schema import DatasetSchema
from anycam4d.enums import CameraModelType
from tests.prep_writing import write_views
from anycam4d.math.transforms import quat_translation_to_c2w


def _c2w(
    translation: tuple[float, float, float] = (0.0, 0.0, 3.0),
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    c2w[:3, 3] = translation
    return c2w


def _make_image(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_mask(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([0, 255], dtype=np.uint8), size=(h, w))


def _make_view(
    cam_id: int = 0,
    frame_id: int = 0,
    timestamp: float = 0.0,
    model: CameraModelType = CameraModelType.PINHOLE,
    params: list[float] | None = None,
    mask: np.ndarray | None = None,
    reproject_source: dict | None = None,
    c2w: np.ndarray | None = None,
) -> CalibratedView:
    h, w = 8, 8
    K = np.array([[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return CalibratedView(
        cam_id=cam_id,
        frame_id=frame_id,
        timestamp=timestamp,
        image=_make_image(h, w, cam_id * 100 + frame_id),
        mask=mask,
        model=model,
        K=K,
        params=params if params is not None else [50.0, 50.0, 4.0, 4.0],
        c2w=c2w if c2w is not None else _c2w(translation=(cam_id * 0.5, 0.0, 3.0)),
        reproject_source=reproject_source,
    )


@pytest.mark.parametrize(
    ("model", "params"),
    [
        (CameraModelType.PINHOLE, [50.0, 50.0, 4.0, 4.0]),
        (CameraModelType.OPENCV, [50.0, 50.0, 4.0, 4.0, 0.1, 0.05, 0.01, 0.02, 0.03]),
        (CameraModelType.OPENCV_FULL, [50.0, 50.0, 4.0, 4.0, *([0.01] * 12)]),
        (CameraModelType.FISHEYE_OPENCV, [50.0, 50.0, 4.0, 4.0, 0.1, 0.05, 0.01, 0.02]),
    ],
)
def test_model_param_round_trip(
    tmp_path: Path,
    model: CameraModelType,
    params: list[float],
) -> None:
    out_dir = tmp_path / model.value
    writer = PrepWriter(out_dir)
    write_views(writer, [_make_view(model=model, params=params)])
    writer.finalize()

    raw = json.loads((out_dir / "train_dataset.json").read_text())
    schema = DatasetSchema.model_validate(raw)
    assert schema.cameras[0].model is model
    assert schema.cameras[0].params == pytest.approx(params)

    dataset = MultiViewSequenceDataset(out_dir)
    assert len(dataset) == 1
    assert dataset.camera_model is model


def test_reproject_source_round_trips_through_the_manifest(tmp_path: Path) -> None:
    """A reprojected-pinhole camera keeps its native lens in the manifest, all the way to the loader."""
    reproject_source = {
        "model": "fisheye_opencv",
        "width": 16,
        "height": 12,
        "params": [30.0, 30.0, 8.0, 6.0, 0.01, -0.005, 0.001, 0.0],
    }
    writer = PrepWriter(tmp_path)
    write_views(writer, [_make_view(model=CameraModelType.PINHOLE, reproject_source=reproject_source)])
    writer.finalize()

    raw = json.loads((tmp_path / "train_dataset.json").read_text())
    assert raw["cameras"][0]["reproject_source"] == reproject_source
    schema = DatasetSchema.model_validate(raw)
    assert schema.cameras[0].reproject_source.model is CameraModelType.FISHEYE_OPENCV

    MultiViewSequenceDataset(tmp_path)


def test_native_pinhole_carries_no_reproject_source(tmp_path: Path) -> None:
    """A camera that was never reprojected stays clean: the provenance key is absent, not null."""
    writer = PrepWriter(tmp_path)
    write_views(writer, [_make_view(model=CameraModelType.PINHOLE)])
    writer.finalize()

    raw = json.loads((tmp_path / "train_dataset.json").read_text())
    assert "reproject_source" not in raw["cameras"][0]


def test_param_length_out_of_range_raises(tmp_path: Path) -> None:
    writer = PrepWriter(tmp_path)
    view = _make_view(model=CameraModelType.PINHOLE, params=[50.0, 50.0, 4.0])
    with pytest.raises(ValueError, match="params"):
        write_views(writer, [view])


def test_pose_round_trip(tmp_path: Path) -> None:
    c2w = _c2w(translation=(1.0, -2.0, 5.0), euler_deg=(12.0, -30.0, 45.0))
    writer = PrepWriter(tmp_path)
    write_views(writer, [_make_view(c2w=c2w)])
    writer.finalize()

    raw = json.loads((tmp_path / "train_dataset.json").read_text())
    cap = raw["captures"][0]
    reconstructed = quat_translation_to_c2w(cap["rotation"], cap["translation"]).numpy()
    np.testing.assert_allclose(reconstructed, c2w, atol=1e-5)


def test_eval_split_routes_by_camera(tmp_path: Path) -> None:
    writer = PrepWriter(tmp_path, eval_cams={1})
    write_views(
        writer,
        [
            _make_view(cam_id=0, frame_id=0, timestamp=0.0),
            _make_view(cam_id=1, frame_id=0, timestamp=0.0),
        ],
    )
    points = np.zeros((5, 3), dtype=np.float32)
    writer.finalize(points=points)

    train_raw = json.loads((tmp_path / "train_dataset.json").read_text())
    eval_raw = json.loads((tmp_path / "eval_dataset.json").read_text())

    assert {c["camera_id"] for c in train_raw["captures"]} == {0}
    assert {c["camera_id"] for c in eval_raw["captures"]} == {1}
    assert train_raw["pointcloud_path"] == "points3d.ply"
    assert "pointcloud_path" not in eval_raw

    DatasetSchema.model_validate(train_raw)
    DatasetSchema.model_validate(eval_raw)
    assert len(MultiViewSequenceDataset(tmp_path, split="train")) == 1
    assert len(MultiViewSequenceDataset(tmp_path, split="eval")) == 1


def test_webp_alpha_stays_exact(tmp_path: Path) -> None:
    # Lossy webp (q95) still encodes the alpha plane losslessly, so a binary mask round-trips exactly.
    mask = _make_mask(8, 8, seed=99)
    writer = PrepWriter(tmp_path, image_format="webp")
    write_views(writer, [_make_view(mask=mask)])
    writer.finalize()

    raw = json.loads((tmp_path / "train_dataset.json").read_text())
    image_path = tmp_path / raw["captures"][0]["image_path"]
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("RGBA"))
    np.testing.assert_array_equal(arr[..., 3], mask)


def test_write_image_skips_existing_for_resume(tmp_path: Path) -> None:
    """An image already on disk is kept (resume) — a re-run never re-encodes finished frames."""
    writer = PrepWriter(tmp_path, image_format="webp")
    write_views(writer, [_make_view()])
    image_path = tmp_path / "images" / "cam00_0000.jpg"  # opaque view -> jpg
    assert image_path.exists()
    image_path.write_bytes(b"SENTINEL")  # stand in for an already-encoded frame

    write_views(PrepWriter(tmp_path, image_format="webp"), [_make_view()])
    assert image_path.read_bytes() == b"SENTINEL"  # untouched


def test_opaque_masked_view_downgrades_to_jpg(tmp_path: Path) -> None:
    """A webp-configured view whose mask is fully valid (no transparency) is written as fast jpg."""
    opaque = np.full((8, 8), 255, dtype=np.uint8)
    write_views(PrepWriter(tmp_path, image_format="webp"), [_make_view(mask=opaque)])
    assert (tmp_path / "images" / "cam00_0000.jpg").exists()
    assert not (tmp_path / "images" / "cam00_0000.webp").exists()


def test_transparent_mask_keeps_webp(tmp_path: Path) -> None:
    """A view with any transparent pixel keeps the alpha-capable webp format."""
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[0, 0] = 0  # one transparent pixel -> needs alpha
    write_views(PrepWriter(tmp_path, image_format="webp"), [_make_view(mask=mask)])
    assert (tmp_path / "images" / "cam00_0000.webp").exists()


def test_transparent_mask_png_preserves_binary_alpha(tmp_path: Path) -> None:
    """png (an alpha carrier) stores an exact binary alpha and is lossless."""
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[0, 0] = 0
    write_views(PrepWriter(tmp_path, image_format="png"), [_make_view(mask=mask)])
    image_path = tmp_path / "images" / "cam00_0000.png"
    assert image_path.exists()
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("RGBA"))
    np.testing.assert_array_equal(arr[..., 3], mask)


def test_jpg_with_mask_raises(tmp_path: Path) -> None:
    """jpg cannot hold an alpha mask -> writing a masked view as jpg fails loudly."""
    mask = _make_mask(8, 8, seed=1)
    writer = PrepWriter(tmp_path, image_format="jpg")
    with pytest.raises(ValueError, match="jpg cannot store"):
        write_views(writer, [_make_view(mask=mask)])


def test_jpg_without_mask_writes(tmp_path: Path) -> None:
    """A maskless view writes fine as jpg and reloads through the dataset."""
    writer = PrepWriter(tmp_path, image_format="jpg")
    write_views(writer, [_make_view()])
    writer.finalize()

    raw = json.loads((tmp_path / "train_dataset.json").read_text())
    assert raw["captures"][0]["image_path"].endswith(".jpg")
    assert len(MultiViewSequenceDataset(tmp_path)) == 1


def test_no_points_omits_pointcloud(tmp_path: Path) -> None:
    writer = PrepWriter(tmp_path)
    write_views(writer, [_make_view()])
    writer.finalize(points=None)

    raw = json.loads((tmp_path / "train_dataset.json").read_text())
    assert "pointcloud_path" not in raw
    assert not (tmp_path / "points3d.ply").exists()


def test_manifest_written(tmp_path: Path) -> None:
    writer = PrepWriter(tmp_path)
    write_views(writer, [_make_view()])
    manifest = build_manifest(source="synthetic", importer="test_prep_writer", cli_args={"foo": "bar"})
    writer.finalize(manifest=manifest)

    raw = json.loads((tmp_path / "prep_manifest.json").read_text())
    assert raw["source"] == "synthetic"
    assert raw["importer"] == "test_prep_writer"
    assert "tools" in raw
    assert "created_utc" in raw


def test_incremental_writes(tmp_path: Path) -> None:
    writer = PrepWriter(tmp_path)
    write_views(
        writer,
        [
            _make_view(cam_id=0, frame_id=0, timestamp=0.0),
            _make_view(cam_id=1, frame_id=0, timestamp=0.0),
        ],
    )
    write_views(
        writer,
        [
            _make_view(cam_id=0, frame_id=1, timestamp=1.0 / 30.0),
            _make_view(cam_id=1, frame_id=1, timestamp=1.0 / 30.0),
        ],
    )
    writer.finalize()

    dataset = MultiViewSequenceDataset(tmp_path)
    assert len(dataset) == 4


def test_finalize_keeps_per_camera_timestamps(tmp_path: Path) -> None:
    """A per-camera time offset within a frame (e.g. SelfCap's sub-frame sync) is written as is."""
    writer = PrepWriter(tmp_path, image_format="png")
    c2w = np.eye(4, dtype=np.float32)
    for cam_id in (0, 1):
        writer.register_camera(cam_id, CameraModelType.PINHOLE, [10.0, 10.0, 4.0, 4.0], 8, 8)
        for frame_id in range(2):
            offset = 0.01 * cam_id
            writer.add_capture(
                cam_id, frame_id, frame_id / 30.0 - offset, c2w, f"images/cam{cam_id:02d}_{frame_id:04d}.png"
            )
    writer.finalize()

    stamps = {
        (c["camera_id"], c["frame_id"]): c["timestamp"]
        for c in json.loads((tmp_path / "train_dataset.json").read_text())["captures"]
    }
    assert stamps[(1, 0)] == pytest.approx(-0.01)


def test_finalize_rejects_a_ragged_rig(tmp_path: Path) -> None:
    """The C>1 batch path indexes a fixed camera count per frame; a frame missing a camera has none."""
    writer = PrepWriter(tmp_path, image_format="png")
    c2w = np.eye(4, dtype=np.float32)
    for cam_id in (0, 1):
        writer.register_camera(cam_id, CameraModelType.PINHOLE, [10.0, 10.0, 4.0, 4.0], 8, 8)
    writer.add_capture(0, 0, 0.0, c2w, "images/cam00_0000.png")
    writer.add_capture(1, 0, 0.0, c2w, "images/cam01_0000.png")
    writer.add_capture(0, 1, 1 / 30.0, c2w, "images/cam00_0001.png")  # camera 1 absent from frame 1

    with pytest.raises(ValueError, match="differing camera counts"):
        writer.finalize()


def test_finalize_accepts_frames_sharing_a_timestamp_across_splits(tmp_path: Path) -> None:
    """Train and eval cameras are disjoint but observe the same frames — checked together, not apart."""
    writer = PrepWriter(tmp_path, eval_cams=[1], image_format="png")
    c2w = np.eye(4, dtype=np.float32)
    for cam_id in (0, 1):
        writer.register_camera(cam_id, CameraModelType.PINHOLE, [10.0, 10.0, 4.0, 4.0], 8, 8)
        for frame_id in range(2):
            writer.add_capture(cam_id, frame_id, frame_id / 30.0, c2w, f"images/cam{cam_id:02d}_{frame_id:04d}.png")

    writer.finalize()

    assert (tmp_path / "train_dataset.json").exists()
    assert (tmp_path / "eval_dataset.json").exists()


def test_finalize_accepts_shared_timestamps(tmp_path: Path) -> None:
    writer = PrepWriter(tmp_path, image_format="png")
    c2w = np.eye(4, dtype=np.float32)
    for cam_id in (0, 1):
        writer.register_camera(cam_id, CameraModelType.PINHOLE, [10.0, 10.0, 4.0, 4.0], 8, 8)
        for frame_id in range(2):
            writer.add_capture(cam_id, frame_id, frame_id / 30.0, c2w, f"images/cam{cam_id:02d}_{frame_id:04d}.png")

    writer.finalize()

    assert (tmp_path / "train_dataset.json").exists()
