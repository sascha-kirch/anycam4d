"""Pydantic schema and validation for the MultiViewSequence dataset format."""

from __future__ import annotations

import math
from pathlib import Path  # noqa: TC003 — used at runtime in pydantic models
from typing import Annotated

from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator

from anycam4d.camera.registry import param_count_range
from anycam4d.enums import CameraModelType  # noqa: TC001 — used at runtime in pydantic models


def _check_max_incidence(value: float | None) -> float | None:
    """A field limit is an incidence angle: strictly positive, and at most the half turn behind."""
    if value is not None and not (0.0 < value <= 180.0):
        raise ValueError(f"max_incidence_deg must be an incidence angle in (0, 180] degrees, got {value}")
    return value


#: A lens's maximum observed incidence angle in degrees, or ``None`` when nothing recorded it.
#:
#: Capture metadata, not a property of the distortion polynomial. A lens's field ends where its
#: optics and its housing end, which ``k1..k4`` does not encode: a circular fisheye's radial map is
#: typically monotone far past its own image circle, so on the polynomial alone the frame corners
#: come out as observations the capture never made. ``None`` asserts nothing more is known, and the
#: monotone domain stands alone — which is exact for a lens that fills its frame.
MaxIncidenceDeg = Annotated[float | None, AfterValidator(_check_max_incidence)]


class ReprojectSourceSchema(BaseModel):
    """Provenance: the native lens a pinhole camera was reprojected (undistorted) from.

    Undistorting a distorted view onto a clean pinhole plane discards the native model — the
    manifest afterwards says "plain pinhole, zero distortion", for which every pixel looks
    observed. But the remap can pull target pixels from outside the source image, and those
    pixels were never observed. Recording the native model, its image dimensions, and its
    intrinsics/distortion makes that footprint replayable: unproject the target grid to rays,
    project them through the native lens, and keep the ones that land inside ``width x height``.
    """

    model: CameraModelType
    width: int
    height: int
    params: list[float]  # [fx, fy, cx, cy, dist...] of the native lens, validated by its own model

    @model_validator(mode="after")
    def _check_params_count(self) -> ReprojectSourceSchema:
        lo, hi = param_count_range(self.model)
        n = len(self.params)
        if not (lo <= n <= hi):
            raise ValueError(f"reproject_source model '{self.model.value}' expects {lo}..{hi} params, got {n}")
        return self


class CameraSchema(BaseModel):
    """Intrinsic calibration for one physical camera."""

    camera_id: int
    model: CameraModelType
    width: int
    height: int
    params: list[float]
    # Optional on any camera, unvalidated by design: a bare PINHOLE entry is genuinely ambiguous
    # (native pinhole vs. reprojected pinhole) and nothing else in the manifest distinguishes them
    # — which is exactly why this block exists. A genuine native pinhole leaves it None.
    reproject_source: ReprojectSourceSchema | None = None
    max_incidence_deg: MaxIncidenceDeg = None

    @model_validator(mode="after")
    def _check_params_count(self) -> CameraSchema:
        # Range check (not exact): 4 intrinsics are mandatory; distortion is any positional
        # OpenCV subset up to the model's max. The canonical slot order == OpenCV distCoeffs
        # order, so exceeding the max is the only illegal positional case (use a higher model).
        lo, hi = param_count_range(self.model)
        n = len(self.params)
        if not (lo <= n <= hi):
            raise ValueError(f"Camera {self.camera_id}: model '{self.model.value}' expects {lo}..{hi} params, got {n}")
        return self


class CaptureSchema(BaseModel):
    """One image observation: path, camera, frame, pose, timestamp."""

    image_path: Path
    camera_id: int
    # Frame identity, shared by every camera that observed this frame — NOT the source video's
    # frame number (that is provenance and lives in prep_manifest.json). Frames are grouped on this
    # integer, never on ``timestamp`` equality: the timestamp is the value rendered at, the frame id
    # is which observations belong together. Conflating them made a per-camera time offset dissolve
    # a 23-camera 600-frame rig into 13800 single-camera frames. Need not be dense or 0-based;
    # consumers sort the unique ids, so a 1-based or gapped numbering (dropped frames) is fine.
    frame_id: int
    timestamp: float
    rotation: list[float]  # [qw, qx, qy, qz] — camera-to-world, OpenCV convention
    translation: list[float]  # [tx, ty, tz] — camera origin in world coords

    @field_validator("rotation")
    @classmethod
    def _check_rotation(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError(f"rotation must have 4 components [qw, qx, qy, qz], got {len(v)}")
        norm = math.sqrt(sum(x * x for x in v))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"rotation quaternion must be unit-norm, got norm={norm:.6f}")
        return v

    @field_validator("translation")
    @classmethod
    def _check_translation(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError(f"translation must have 3 components [tx, ty, tz], got {len(v)}")
        return v


class MetadataSchema(BaseModel):
    """Dataset-level coordinate convention metadata."""

    world_coordinate_system: str
    camera_coordinate_system: str
    transform_convention: str


class DatasetSchema(BaseModel):
    """Top-level schema for train_dataset.json / eval_dataset.json."""

    schema_version: int
    metadata: MetadataSchema
    pointcloud_path: Path | None = None
    # Real-time span mapped to tau in [-0.5, 0.5]; None means the captures' own min..max.
    time_range: tuple[float, float] | None = None
    cameras: list[CameraSchema]
    captures: list[CaptureSchema]

    @model_validator(mode="after")
    def _check_camera_id_refs(self) -> DatasetSchema:
        valid_ids = {c.camera_id for c in self.cameras}
        for i, cap in enumerate(self.captures):
            if cap.camera_id not in valid_ids:
                raise ValueError(f"Capture {i}: camera_id={cap.camera_id} not in cameras (valid: {sorted(valid_ids)})")
        return self

    @model_validator(mode="after")
    def _check_time_range(self) -> DatasetSchema:
        if self.time_range is None:
            return self
        t0, t1 = self.time_range
        outside = [cap.timestamp for cap in self.captures if not (t0 <= cap.timestamp <= t1)]
        if t0 >= t1 or outside:
            raise ValueError(f"time_range {self.time_range} must be increasing and hold every timestamp: {outside[:4]}")
        return self
