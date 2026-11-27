"""RenderSettings: the live render settings the panels write and the render adapter reads."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Literal

from anycam4d.enums import CameraModelType


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


@dataclasses.dataclass
class RenderSettings:
    """Live render settings written by the GUI panels and read by the render adapter.

    Args:
        render_mode: Output channel — ``"rgb"``, ``"alpha"``, or ``"temporal"``.
        background_color: RGB background in [0, 1], shown where alpha < 1.
        colormap: Colormap used for the alpha / temporal visualizations.
        temporal_color_mode: How temporal mode colors Gaussians — ``"mean"`` (by mean
            time ``mu_t``), ``"distance"`` (by ``|tau - mu_t|`` at the current time), or
            ``"std"`` (by the marginal temporal width ``std_t_marg``, what the gate divides by).
        active_camera_model: Projection model used by the rasterizer.
        active_distortion_params: Canonical distortion vector for the active model, or None for a plain pinhole.
        fisheye_field_deg: Angle a fisheye images; pixels past it show the background.
        fisheye_circular: The fisheye field fits inside the frame (circular) rather than spanning its diagonal.
    """

    render_mode: Literal["rgb", "alpha", "temporal"] = "rgb"
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    colormap: Literal["turbo", "viridis", "magma", "inferno", "cividis", "gray"] = "turbo"
    temporal_color_mode: Literal["mean", "distance", "std"] = "mean"
    active_camera_model: CameraModelType = CameraModelType.PINHOLE
    active_distortion_params: Float[Tensor, "..."] | None = None
    fisheye_field_deg: float = 180.0
    fisheye_circular: bool = True
