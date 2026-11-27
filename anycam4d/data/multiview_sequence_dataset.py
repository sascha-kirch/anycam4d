"""MultiViewSequenceDataset — reads train_dataset.json / eval_dataset.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from anycam4d.camera.validity import intrinsics_from_params
from anycam4d.camera.validity import resolve_validity
from anycam4d.data.normalization import normalize_viewmat
from anycam4d.data.schema import DatasetSchema
from anycam4d.data.types import TrainingSample
from anycam4d.math.transforms import quat_translation_to_c2w


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.data.schema import CameraSchema
    from anycam4d.enums import CameraModelType


class MultiViewSequenceDataset(torch.utils.data.Dataset):
    """Dataset over ``train_dataset.json`` / ``eval_dataset.json``; images load lazily in ``__getitem__``.

    Poses are camera-to-world quaternion + translation in OpenCV convention.

    Args:
        data_dir: Dataset root.
        split: ``"train"`` or ``"eval"``.
        scale_factor: Integer downscale applied to intrinsics and images.
    """

    def __init__(  # noqa: PLR0915, C901, PLR0912 — linear per-capture load; splitting would only scatter it
        self,
        data_dir: str | Path,
        split: str = "train",
        scale_factor: int = 1,
    ) -> None:
        """Load dataset index from JSON, pre-computing viewmats and intrinsics."""
        self.data_dir = Path(data_dir)
        self.scale_factor = scale_factor

        if split == "train":
            filename = "train_dataset.json"
        elif split == "eval":
            filename = "eval_dataset.json"
        else:
            raise ValueError(f"Got unsupported split: '{split}'. Valid: [train|eval]")

        json_path = self.data_dir / filename
        with json_path.open() as f:
            raw = json.load(f)
        schema = DatasetSchema.model_validate(raw)

        cameras_by_id = {cam.camera_id: cam for cam in schema.cameras}

        times = [cap.timestamp for cap in schema.captures]
        t_min, t_max = schema.time_range or (min(times), max(times))
        self._t_min = float(t_min)
        self._t_max = float(t_max)
        t_range = self._t_max - self._t_min
        self._timestamps = [(t - self._t_min) / t_range - 0.5 if t_range > 0.0 else 0.0 for t in times]

        sf = float(scale_factor)
        self._viewmats: list[Float[Tensor, "4 4"]] = []
        self._Ks: list[Float[Tensor, "3 3"]] = []
        self._image_paths: list[Path] = []
        self._camera_ids: list[int] = []
        self._camera_models: list[CameraModelType] = []
        self._distortion_params_list: list[Float[Tensor, "P"]] = []

        for cap in schema.captures:
            cam = cameras_by_id[cap.camera_id]
            c2w = quat_translation_to_c2w(cap.rotation, cap.translation)
            self._viewmats.append(torch.linalg.inv(c2w))

            K, dist = intrinsics_from_params(cam.model, cam.params, sf)
            self._Ks.append(K)
            self._image_paths.append(self.data_dir / cap.image_path)
            self._camera_ids.append(cap.camera_id)  # raw ids; remapped to 0-based contiguous below
            self._camera_models.append(cam.model)
            self._distortion_params_list.append(dist)

        # Remap camera ids to 0..n-1; the bilateral grid indexes one grid per id.
        id_remap = {raw: i for i, raw in enumerate(sorted(set(self._camera_ids)))}
        self._camera_ids = [id_remap[raw] for raw in self._camera_ids]

        # The manifest entry of every camera this split actually captures, keyed by remapped id.
        # Single source of truth for per-camera (as opposed to per-capture) facts.
        self._camera_schema_by_id: dict[int, CameraSchema] = {
            new_id: cameras_by_id[raw_id] for raw_id, new_id in id_remap.items()
        }

        self._validity: dict[int, Bool[Tensor, "h w"]] = self._resolve_validity()
        # Decided dataset-wide, so a mixed rig never drops a partially observed camera's mask.
        self._all_trivially_valid = all(bool(v.all()) for v in self._validity.values())

        # Frames group on frame_id, never on timestamp equality; ids may be sparse, so the frame index
        # is the rank of the sorted unique ids.
        self._frame_to_indices: dict[int, list[int]] = defaultdict(list)
        self._frame_id_by_index: list[int] = []
        for idx, cap in enumerate(schema.captures):
            self._frame_to_indices[cap.frame_id].append(idx)
            self._frame_id_by_index.append(cap.frame_id)
        self._frame_ids: list[int] = sorted(self._frame_to_indices.keys())

        # cameras_per_timestep and the C>1 frame-aligned path assume a uniform camera count
        # across all frames. Validate once at load and fail loudly otherwise (fail-fast style).
        frame_sizes = {fid: len(idxs) for fid, idxs in self._frame_to_indices.items()}
        unique_sizes = set(frame_sizes.values())
        if len(unique_sizes) > 1:
            ragged = sorted(fid for fid, n in frame_sizes.items() if n != max(unique_sizes))[:5]
            raise ValueError(
                f"Non-uniform camera count across frames (got sizes {sorted(unique_sizes)}; "
                f"e.g. frames {ragged}). All frames must share the same number of cameras."
            )

        # Cameras of a frame may be offset in time (each capture renders at its own tau); a frame's tau is their mean.
        self._unique_timestamps: list[float] = []
        self._frame_synchronous = True
        for fid in self._frame_ids:
            taus = [self._timestamps[idx] for idx in self._frame_to_indices[fid]]
            self._frame_synchronous &= max(taus) - min(taus) <= 1e-8
            self._unique_timestamps.append(sum(taus) / len(taus))

        self._pointcloud_path: Path | None = None
        if schema.pointcloud_path is not None:
            p = self.data_dir / schema.pointcloud_path
            if p.exists():
                self._pointcloud_path = p
        else:
            for name in ("point_cloud.ply", "points3d.ply", "sparse_pc.ply"):
                p = self.data_dir / name
                if p.exists():
                    self._pointcloud_path = p
                    break

    def _resolve_validity(self) -> dict[int, Bool[Tensor, "h w"]]:
        """Validity mask per 0-based camera id at the working resolution, from rescaled intrinsics."""
        validity: dict[int, Bool[Tensor, "h w"]] = {}
        for cam_id, cam in self._camera_schema_by_id.items():
            w_work = cam.width // self.scale_factor
            h_work = cam.height // self.scale_factor
            on_disk = self._load_valid_mask(cam, w_work, h_work)
            validity[cam_id] = resolve_validity(cam, on_disk, w_work, h_work)
        return validity

    def _load_valid_mask(
        self,
        camera: CameraSchema,
        width: int,
        height: int,
    ) -> Bool[Tensor, "h w"] | None:
        """``valid/<manifest camera_id>.<ext>`` as a bool mask, or ``None``; several matches raise ``FileNotFoundError``."""
        matches = sorted((self.data_dir / "valid").glob(f"{camera.camera_id}.*"))
        if not matches:
            return None
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Ambiguous validity mask for camera {camera.camera_id}: {[p.name for p in matches]}"
            )

        with Image.open(matches[0]) as pil_mask:
            arr = np.asarray(pil_mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = torch.from_numpy(arr > 0)

        if mask.shape != (height, width):
            # Nearest only — a validity mask is categorical, so area interpolation would invent
            # partially-valid pixels that mean nothing.
            resized = F.interpolate(mask[None, None].float(), size=(height, width), mode="nearest")
            mask = resized[0, 0] > 0.5
        return mask

    def __len__(self) -> int:
        """Number of captures, not timesteps."""
        return len(self._timestamps)

    def __getitem__(self, idx: int) -> TrainingSample:
        with Image.open(self._image_paths[idx]) as pil_img:
            # Normalize to either RGB or RGBA. This handles palette, grayscale, etc.
            mode = "RGBA" if "A" in pil_img.getbands() else "RGB"
            arr = np.asarray(pil_img.convert(mode)).astype(np.float32) / 255.0

        image = torch.from_numpy(arr[..., :3])

        mask = torch.from_numpy(arr[..., 3:4]) if arr.shape[-1] == 4 else None

        if self.scale_factor > 1:
            image, mask = _resize_sample(image, mask, self.scale_factor)

        # Validity was resolved at the working resolution at init, so it never goes through
        # ``_resize_sample`` — only the image and the semantic alpha are downscaled per sample.
        cam_id = self._camera_ids[idx]
        validity = None if self._all_trivially_valid else self._validity[cam_id].unsqueeze(-1)

        return TrainingSample(
            image=image,
            K=self._Ks[idx],
            viewmat=self._viewmats[idx],
            timestamp=self._timestamps[idx],
            image_id=idx,
            camera_id=cam_id,
            camera_model=self._camera_models[idx],
            distortion_params=self._distortion_params_list[idx],
            mask=mask,
            validity=validity,
        )

    @property
    def timestamps(self) -> list[float]:
        """Per-frame normalized timestamp (mean over the frame's cameras), in frame order."""
        return self._unique_timestamps

    @property
    def frame_synchronous(self) -> bool:
        """Whether every camera of every frame shares one timestamp, as a multi-camera render needs."""
        return self._frame_synchronous

    @property
    def frame_ids(self) -> list[int]:
        """The dataset's own frame ids, sorted. Index ``i`` is what ``indices_at_frame(i)`` refers to."""
        return self._frame_ids

    def frame_id_for(self, index: int) -> int:
        """Capture ``frame_id`` of a flat sample index, independent of sampler order."""
        return self._frame_id_by_index[index]

    @property
    def num_timesteps(self) -> int:
        """Number of frames."""
        return len(self._frame_ids)

    @property
    def cameras_per_timestep(self) -> int:
        """Cameras per frame (uniform, validated at load)."""
        if not self._frame_ids:
            return 0
        return len(self._frame_to_indices[self._frame_ids[0]])

    @property
    def n_cameras(self) -> int:
        """Number of cameras; ids are contiguous from 0."""
        if not self._camera_ids:
            return 0
        return len(set(self._camera_ids))

    @property
    def camera_model(self) -> CameraModelType:
        """Camera model of the first capture; collate enforces one model per batch."""
        return self._camera_models[0]

    def raw_camera_id(self, camera_id: int) -> int | None:
        """Manifest camera id for a 0-based id, or ``None``; use it for anything keyed to the rig."""
        cam = self._camera_schema_by_id.get(camera_id)
        return None if cam is None else int(cam.camera_id)

    @property
    def fully_observed(self) -> bool:
        """Whether every camera observed every pixel, so batches carry no validity."""
        return self._all_trivially_valid

    def validity_for(self, camera_id: int) -> Bool[Tensor, "h w"] | None:
        """Validity mask of a 0-based camera id at the working resolution, or ``None`` if unknown."""
        return self._validity.get(camera_id)

    @property
    def point_cloud_path(self) -> Path | None:
        """Path to the point cloud PLY file, or None if not found."""
        return self._pointcloud_path

    @property
    def t_min(self) -> float:
        """Minimum real-time timestamp (seconds) before normalization."""
        return self._t_min

    @property
    def t_max(self) -> float:
        """Maximum real-time timestamp (seconds) before normalization."""
        return self._t_max

    @property
    def viewmats(self) -> list[Float[Tensor, "4 4"]]:
        """Per-capture world-to-camera view matrices."""
        return self._viewmats

    @property
    def camera_ids(self) -> list[int]:
        """0-based camera id per index, without image I/O."""
        return self._camera_ids

    def indices_at_frame(self, frame_idx: int) -> list[int]:
        """Dataset indices for all cameras of the ``frame_idx``-th frame (rank, not the frame id)."""
        return self._frame_to_indices[self._frame_ids[frame_idx]]

    def apply_viewmat_transform(self, T: Float[Tensor, "4 4"]) -> None:
        """Move every stored viewmat into the normalized space of ``T`` (``T_denorm``) in place, kept rigid."""
        for i in range(len(self._viewmats)):
            self._viewmats[i] = normalize_viewmat(self._viewmats[i], T)


def _resize_sample(
    image: Float[Tensor, "H W 3"],
    mask: Float[Tensor, "H W 1"] | None,
    scale_factor: int,
) -> tuple[Float[Tensor, "H W 3"], Float[Tensor, "H W 1"] | None]:
    """Downscale image and mask by scale_factor using area/nearest interpolation."""
    h, w = image.shape[0], image.shape[1]
    new_h, new_w = h // scale_factor, w // scale_factor

    img_chw = image.permute(2, 0, 1).unsqueeze(0)
    img_chw = F.interpolate(img_chw, size=(new_h, new_w), mode="area")
    image = img_chw.squeeze(0).permute(1, 2, 0)

    if mask is not None:
        mask_nchw = mask.permute(2, 0, 1).unsqueeze(0)
        mask_nchw = F.interpolate(mask_nchw, size=(new_h, new_w), mode="nearest")
        mask = mask_nchw.squeeze(0).permute(1, 2, 0)

    return image, mask
