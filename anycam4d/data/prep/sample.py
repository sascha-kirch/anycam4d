"""The dataset-prep pivot type: one calibrated capture, host-side and torch-free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import Float
    from jaxtyping import UInt8

    from anycam4d.enums import CameraModelType


@dataclass(frozen=True)
class CalibratedView:
    """A 1:1 in-memory image of one AnyCam4D capture.

    The native pivot every importer, ``Stage``, and writer in ``anycam4d/data/prep`` exchanges.
    Numpy/host-side by design — distinct from ``anycam4d.data.types.TrainingSample``, which is
    the torch, train-time sample produced after normalization and loading.
    """

    cam_id: int
    frame_id: int
    timestamp: float  # seconds, pre-normalization
    image: UInt8[np.ndarray, "H W 3"]
    mask: UInt8[np.ndarray, "H W"] | None
    model: CameraModelType
    K: Float[np.ndarray, "3 3"]
    params: list[float]  # [fx, fy, cx, cy, <distortion...>], canonical OpenCV order
    c2w: Float[np.ndarray, "4 4"]  # OpenCV camera-to-world
    # Provenance when this view was resampled off a different (native) lens onto a clean pinhole:
    # {"model", "width", "height", "params"} of the *pre-reprojection* camera. Undistorting discards
    # the native model, after which every pixel of the stored pinhole looks observed even though the
    # remap pulls some from outside the source image; this block keeps that footprint replayable.
    reproject_source: dict | None = None
    # The lens's maximum observed incidence angle in degrees, or ``None`` when nothing recorded it
    # and nothing measured it. Capture metadata rather than calibration: a circular lens's radial
    # map stays monotone past its own image circle, so a stage that fits a grid or a loader that
    # computes validity has no way to find the rim without it.
    max_incidence_deg: float | None = None
