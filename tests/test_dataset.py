"""``MultiViewSequenceDataset`` and samplers: loading, timestamps, formats, splits, frames and observation validity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from anycam4d.camera.validity import resolve_validity
from anycam4d.config.training import TrainingConfig
from anycam4d.data.factory import DatasetFactory
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.samplers import FrameAlignedSampler
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.enums import SamplerType


# A strongly barrel-distorted OpenCV lens whose frame corners fall outside the modelled domain
# (~65% of pixels observed at 64x48), paired with a distortion-free entry of the same model so a
# mixed dataset stays collatable (anycam4d_collate rejects mixed camera models).
_W, _H = 64, 48
_DISTORTED_PARAMS = [35.0, 35.0, 32.0, 24.0, -0.3]
_UNDISTORTED_PARAMS = [35.0, 35.0, 32.0, 24.0]


def _write_validity_dataset(base: Path, params_per_camera: list[list[float]]) -> Path:
    """Write a 1-frame MultiViewSequence dataset with one OPENCV camera per parameter vector."""
    (base / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)

    cameras = [
        {"camera_id": i, "model": "opencv", "width": _W, "height": _H, "params": params}
        for i, params in enumerate(params_per_camera)
    ]
    captures = []
    for i in range(len(params_per_camera)):
        rgba = np.concatenate(
            [
                rng.integers(0, 256, (_H, _W, 3), dtype=np.uint8),
                rng.choice(np.array([0, 255], dtype=np.uint8), size=(_H, _W, 1)),
            ],
            axis=2,
        )
        Image.fromarray(rgba, mode="RGBA").save(base / "images" / f"cam{i:02d}_0000.png")
        captures.append(
            {
                "image_path": f"images/cam{i:02d}_0000.png",
                "camera_id": i,
                "frame_id": 0,
                "timestamp": 0.0,
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [i * 0.5, 0.0, 3.0],
            }
        )

    schema = {
        "schema_version": 1,
        "metadata": {
            "world_coordinate_system": "right_handed",
            "camera_coordinate_system": "opencv",
            "transform_convention": "camera_to_world",
        },
        "cameras": cameras,
        "captures": captures,
    }
    with open(base / "train_dataset.json", "w") as f:
        json.dump(schema, f)
    return base


@pytest.fixture(scope="module")
def synth_dataset(tmp_dataset: Path) -> MultiViewSequenceDataset:
    """MultiViewSequenceDataset from the synthetic fixture (2 frames × 3 cameras)."""
    return MultiViewSequenceDataset(tmp_dataset, split="train")


@pytest.fixture(scope="module")
def distorted_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dataset whose two cameras both have genuinely unobserved corners."""
    return _write_validity_dataset(tmp_path_factory.mktemp("validity_distorted"), [_DISTORTED_PARAMS] * 2)


@pytest.fixture(scope="module")
def mixed_validity_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dataset mixing a fully-observed camera (cam 0) with a partially-observed one (cam 1)."""
    return _write_validity_dataset(tmp_path_factory.mktemp("validity_mixed"), [_UNDISTORTED_PARAMS, _DISTORTED_PARAMS])


def test_dataset_loads_all_frames(synth_dataset: MultiViewSequenceDataset) -> None:
    """Every __getitem__ call succeeds and returns a TrainingSample."""
    assert len(synth_dataset) == 6  # 2 frames × 3 cameras

    for idx in range(len(synth_dataset)):
        sample = synth_dataset[idx]
        assert sample.image is not None
        assert sample.K is not None
        assert sample.viewmat is not None


def test_timestamps_normalized(synth_dataset: MultiViewSequenceDataset) -> None:
    """All normalized timestamps are in [-0.5, 0.5] (original 4DGS convention)."""
    for idx in range(len(synth_dataset)):
        sample = synth_dataset[idx]
        assert -0.5 - 1e-6 <= sample.timestamp <= 0.5 + 1e-6, f"idx={idx}: timestamp={sample.timestamp}"

    for tau in synth_dataset.timestamps:
        assert -0.5 - 1e-6 <= tau <= 0.5 + 1e-6


def test_timestamp_denormalization() -> None:
    """Denormalization from [-0.5, 0.5] back to real time is exact."""
    tau = 0.25
    t_min, t_max = 0.0, 1.0
    t_real = t_min + (tau + 0.5) * (t_max - t_min)  # = 0.75
    assert abs(t_real - 0.75) < 1e-6


def test_timestamps_boundary_values(synth_dataset: MultiViewSequenceDataset) -> None:
    """First and last unique timestamps are ≈ -0.5 and ≈ +0.5 respectively."""
    taus = synth_dataset.timestamps
    assert abs(taus[0] - (-0.5)) < 1e-6, f"First timestamp expected ≈-0.5, got {taus[0]}"
    assert abs(taus[-1] - 0.5) < 1e-6, f"Last timestamp expected ≈+0.5, got {taus[-1]}"


def test_rgba_split_to_image_and_mask(synth_dataset: MultiViewSequenceDataset) -> None:
    """RGBA images are split; image has 3 channels; mask is in {0.0, 1.0}."""
    for idx in range(len(synth_dataset)):
        sample = synth_dataset[idx]

        assert sample.image.shape[-1] == 3, f"idx={idx}: image has {sample.image.shape[-1]} channels"

        assert sample.mask is not None, f"idx={idx}: mask is None for RGBA image"
        assert sample.mask.shape == (sample.image.shape[0], sample.image.shape[1], 1)

        thresholded = (sample.mask > 0.5).float()
        unique_vals = thresholded.unique().tolist()
        assert all(v in (0.0, 1.0) for v in unique_vals)


def test_webp_lossless_rgba_loads(tmp_dataset_mixed_formats: Path) -> None:
    """Lossless WEBP RGBA loads via the generic pillow loader; mask values in {0, 1}."""
    dataset = MultiViewSequenceDataset(tmp_dataset_mixed_formats, split="train")
    sample = dataset[0]  # cam00_0000.webp

    assert sample.image.shape == (48, 64, 3)
    assert sample.mask is not None
    assert sample.mask.shape == (48, 64, 1)

    thresholded = (sample.mask > 0.5).float()
    unique_vals = thresholded.unique().tolist()
    assert all(v in (0.0, 1.0) for v in unique_vals)


def test_jpeg_rgb_loads_without_mask(tmp_dataset_mixed_formats: Path) -> None:
    """JPEG has no alpha channel; loader must return mask=None, image with 3 channels."""
    dataset = MultiViewSequenceDataset(tmp_dataset_mixed_formats, split="train")
    sample = dataset[1]  # cam01_0000.jpg

    assert sample.image.shape == (48, 64, 3)
    assert sample.mask is None


def test_eval_split_loads(tmp_dataset: Path) -> None:
    """eval_dataset.json loads correctly with fewer captures than train."""
    eval_ds = MultiViewSequenceDataset(tmp_dataset, split="eval")
    assert len(eval_ds) == 2  # fixture: one camera per frame, 2 frames
    sample = eval_ds[0]
    assert sample.image.shape[-1] == 3


def test_frame_aligned_sampler_groups(synth_dataset: MultiViewSequenceDataset) -> None:
    """Every batch from FrameAlignedSampler has all samples at the same timestamp."""
    sampler = FrameAlignedSampler(synth_dataset, shuffle=False)
    batches = list(sampler)
    assert len(batches) == synth_dataset.num_timesteps

    for batch_indices in batches:
        timestamps = [synth_dataset[i].timestamp for i in batch_indices]
        assert len(set(timestamps)) == 1, f"Batch contains mixed timestamps: {timestamps}"
        assert len(batch_indices) == synth_dataset.cameras_per_timestep


def _collate_frame(dataset: MultiViewSequenceDataset, frame_idx: int = 0) -> dict:
    """Collate one full frame (all cameras at one timestamp) the way training does."""
    return anycam4d_collate([dataset[i] for i in dataset.indices_at_frame(frame_idx)])


def test_validity_none_for_fully_observed_dataset(tmp_dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An all-pinhole dataset observes every pixel: no per-sample validity, no warning."""
    dataset = MultiViewSequenceDataset(tmp_dataset, split="train")

    assert dataset[0].validity is None
    assert _collate_frame(dataset)["validity"] is None
    assert "undetermined" not in capsys.readouterr().out


def test_validity_batched_for_distorted_dataset(distorted_dataset: Path) -> None:
    """A lens with unobserved corners yields a full [C, H, W, 1] bool batch matching the provider."""
    dataset = MultiViewSequenceDataset(distorted_dataset, split="train")
    batch = _collate_frame(dataset)
    validity = batch["validity"]

    assert validity is not None
    assert validity.shape == (2, _H, _W, 1)
    assert validity.dtype == torch.bool
    assert not bool(validity.all()), "the distorted fixture must have unobserved pixels"

    for row, cam_id in enumerate(batch["camera_ids"]):
        expected = resolve_validity(dataset._camera_schema_by_id[cam_id], None, _W, _H)
        assert torch.equal(validity[row, ..., 0], expected)


def test_validity_mixed_dataset_keeps_every_camera(mixed_validity_dataset: Path) -> None:
    """A fully-observed camera in the batch must not collapse the partially-observed one to None."""
    dataset = MultiViewSequenceDataset(mixed_validity_dataset, split="train")
    batch = _collate_frame(dataset)
    validity = batch["validity"]

    assert validity is not None, "one trivially-valid camera must not drop the batch to None"
    by_cam = {cam_id: validity[row, ..., 0] for row, cam_id in enumerate(batch["camera_ids"])}
    assert bool(by_cam[0].all()), "the undistorted camera observes its whole frame"
    assert not bool(by_cam[1].all()), "the distorted camera has unobserved corners"


def test_validity_on_disk_mask_wins(tmp_path: Path) -> None:
    """An explicit valid/<camera_id> mask overrides the procedural footprint."""
    base = _write_validity_dataset(tmp_path / "on_disk", [_DISTORTED_PARAMS])
    (base / "valid").mkdir()
    on_disk = np.zeros((_H, _W), dtype=np.uint8)
    on_disk[: _H // 2] = 255  # deliberately unlike the radial footprint
    Image.fromarray(on_disk, mode="L").save(base / "valid" / "0.png")

    dataset = MultiViewSequenceDataset(base, split="train")
    validity = _collate_frame(dataset)["validity"]
    assert validity is not None
    assert torch.equal(validity[0, ..., 0], torch.from_numpy(on_disk > 0))


def test_validity_on_disk_mask_is_nearest_resized(tmp_path: Path) -> None:
    """A full-resolution valid/ mask is nearest-resized to the working resolution."""
    base = _write_validity_dataset(tmp_path / "on_disk_resize", [_DISTORTED_PARAMS])
    (base / "valid").mkdir()
    on_disk = np.zeros((_H, _W), dtype=np.uint8)
    on_disk[: _H // 2] = 255
    Image.fromarray(on_disk, mode="L").save(base / "valid" / "0.png")

    dataset = MultiViewSequenceDataset(base, split="train", scale_factor=2)
    validity = _collate_frame(dataset)["validity"]

    assert validity is not None
    assert validity.shape == (1, _H // 2, _W // 2, 1)
    assert bool(validity[0, : _H // 4, :, 0].all())
    assert not bool(validity[0, _H // 4 :, :, 0].any())


def test_validity_ambiguous_on_disk_mask_raises(tmp_path: Path) -> None:
    """Two files for the same camera id is unresolvable and must fail loudly."""
    base = _write_validity_dataset(tmp_path / "ambiguous", [_DISTORTED_PARAMS])
    (base / "valid").mkdir()
    arr = np.full((_H, _W), 255, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(base / "valid" / "0.png")
    Image.fromarray(arr, mode="L").save(base / "valid" / "0.webp", lossless=True)

    with pytest.raises(FileNotFoundError, match="Ambiguous validity mask"):
        MultiViewSequenceDataset(base, split="train")


def test_validity_resolution_follows_scale_factor(distorted_dataset: Path) -> None:
    """Validity is built at the working resolution and agrees with the scaled intrinsics."""
    full = MultiViewSequenceDataset(distorted_dataset, split="train")
    half = MultiViewSequenceDataset(distorted_dataset, split="train", scale_factor=2)

    validity = _collate_frame(half)["validity"]
    assert validity is not None
    assert validity.shape == (2, _H // 2, _W // 2, 1)

    frac_full = float(_collate_frame(full)["validity"].float().mean())
    frac_half = float(validity.float().mean())
    assert abs(frac_full - frac_half) < 0.02, f"valid fraction moved with resolution: {frac_full} vs {frac_half}"

    # The footprint is computed from the same intrinsics the renderer receives: 64x48 halves
    # evenly, so resolve_validity's internal camera.width/width scaling equals scale_factor and
    # the mask matches the batch's own Ks exactly.
    expected = resolve_validity(half._camera_schema_by_id[0], None, _W // 2, _H // 2)
    assert torch.equal(validity[0, ..., 0], expected)
    assert torch.allclose(half._Ks[0], full._Ks[0] / torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]))


def _write_frame_id_dataset(base: Path, plan: list[tuple[int, float, list[int]]]) -> Path:
    """Write a dataset from ``(frame_id, timestamp, camera_ids)`` triples, images included."""
    (base / "images").mkdir(parents=True, exist_ok=True)
    cam_ids = sorted({cid for _, _, cids in plan for cid in cids})
    captures = []
    for frame_id, timestamp, cids in plan:
        for cid in cids:
            name = f"cam{cid:02d}_{frame_id:04d}.png"
            Image.fromarray(np.zeros((_H, _W, 3), dtype=np.uint8), mode="RGB").save(base / "images" / name)
            captures.append(
                {
                    "image_path": f"images/{name}",
                    "camera_id": cid,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [cid * 0.5, 0.0, 3.0],
                }
            )
    schema = {
        "schema_version": 1,
        "metadata": {
            "world_coordinate_system": "right_handed",
            "camera_coordinate_system": "opencv",
            "transform_convention": "camera_to_world",
        },
        "cameras": [
            {"camera_id": cid, "model": "pinhole", "width": _W, "height": _H, "params": _UNDISTORTED_PARAMS}
            for cid in cam_ids
        ],
        "captures": captures,
    }
    with open(base / "train_dataset.json", "w") as f:
        json.dump(schema, f)
    return base


def test_frames_come_from_frame_id_not_sorted_timestamps(tmp_path: Path) -> None:
    """Frame ids need not be dense or 0-based; the frame *index* is their rank."""
    plan = [(7, 0.0, [0, 1]), (3, -1 / 30.0, [0, 1]), (11, 1 / 30.0, [0, 1])]
    ds = MultiViewSequenceDataset(_write_frame_id_dataset(tmp_path, plan))

    assert ds.num_timesteps == 3
    assert ds.frame_ids == [3, 7, 11]  # sorted ids, gaps and 1-basedness preserved
    assert ds.cameras_per_timestep == 2
    # Frame index 0 is the *lowest id*, which here is also the earliest timestamp.
    assert len(ds.indices_at_frame(0)) == 2
    assert ds.timestamps == pytest.approx([-0.5, 0.0, 0.5])  # normalized, in frame order


def test_time_range_places_a_frame_subset_on_the_full_axis(tmp_path: Path) -> None:
    """A split holding only some frames keeps their tau on the scene's axis, not its own min..max."""
    base = _write_frame_id_dataset(tmp_path, [(1, 1.0, [0]), (2, 2.0, [0])])
    raw = json.loads((base / "train_dataset.json").read_text())
    raw["time_range"] = [0.0, 4.0]
    (base / "train_dataset.json").write_text(json.dumps(raw))

    assert MultiViewSequenceDataset(base).timestamps == pytest.approx([-0.25, 0.0])

    raw["time_range"] = [0.0, 1.5]
    (base / "train_dataset.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="time_range"):
        MultiViewSequenceDataset(base)


def test_frame_id_survives_a_capture_ordering_shuffle(tmp_path: Path) -> None:
    """Captures are grouped by their own id, so JSON ordering carries no meaning."""
    plan = [(0, 0.0, [0, 1]), (1, 1 / 30.0, [0, 1])]
    base = _write_frame_id_dataset(tmp_path, plan)
    raw = json.loads((base / "train_dataset.json").read_text())
    raw["captures"] = list(reversed(raw["captures"]))
    (base / "train_dataset.json").write_text(json.dumps(raw))

    ds = MultiViewSequenceDataset(base)

    assert ds.num_timesteps == 2
    assert ds.cameras_per_timestep == 2


def test_per_camera_timestamps_within_a_frame_load_but_refuse_frame_aligned(tmp_path: Path) -> None:
    """A frame's cameras may be offset in time: each capture keeps its own tau, the frame gets their mean, and only
    the one-pass multi-camera batch refuses them."""
    base = _write_frame_id_dataset(tmp_path, [(0, 0.0, [0, 1]), (1, 1 / 30.0, [0, 1])])
    raw = json.loads((base / "train_dataset.json").read_text())
    offset_cam1 = [cap for cap in raw["captures"] if cap["camera_id"] == 1]
    for cap in offset_cam1:
        cap["timestamp"] += 0.004
    (base / "train_dataset.json").write_text(json.dumps(raw))

    ds = MultiViewSequenceDataset(base)

    assert not ds.frame_synchronous
    taus = sorted({round(ds[i].timestamp, 6) for i in ds.indices_at_frame(0)})
    assert len(taus) == 2
    assert ds.timestamps[0] == pytest.approx(sum(taus) / 2, abs=1e-6)
    with pytest.raises(ValueError, match="offset in time"):
        DatasetFactory.create_dataloader(ds, TrainingConfig(sampler=SamplerType.FRAME_ALIGNED, batch_size=-1))


def test_ragged_frame_raises(tmp_path: Path) -> None:
    plan = [(0, 0.0, [0, 1]), (1, 1 / 30.0, [0])]

    with pytest.raises(ValueError, match="Non-uniform camera count"):
        MultiViewSequenceDataset(_write_frame_id_dataset(tmp_path, plan))
