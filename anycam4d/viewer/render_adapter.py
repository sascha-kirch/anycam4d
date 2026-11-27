"""One camera-math path for interactive and offline rendering, with per-frame caches.

Reads tau and the model from the viewer and the rest from its ``render_settings``; conditioning is not cached.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from nerfview import apply_float_colormap

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.camera.registry import ProjectionFamily
from anycam4d.camera.registry import distortion_width
from anycam4d.camera.validity import compute_validity
from anycam4d.data.normalization import normalize_viewmat
from anycam4d.enums import CameraModelType
from anycam4d.math.covariance import temporal_marginal_from_rot4d
from anycam4d.math.lens_domain import fisheye_domain_radius
from anycam4d.primitives.fourdgaussians.conditioning import temporal_marginal_std
from anycam4d.utils import to_float64_numpy
from anycam4d.viewer.nerfview_adapter import RenderTabView


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from jaxtyping import UInt8
    from nerfview import CameraState
    from nerfview import RenderTabState
    from torch import Tensor

    from anycam4d.renderer.backend import RenderOutput
    from anycam4d.viewer.viewer import Anycam4DViewer


def _is_fisheye(model: CameraModelType) -> bool:
    return CAMERA_REGISTRY[model].family is ProjectionFamily.FISHEYE_FAMILY


class RenderAdapter:
    """Viewer-side render flow: camera math, per-frame caches, render entry points."""

    def __init__(self, viewer: Anycam4DViewer) -> None:
        """Initialize the adapter bound to a viewer; caches start empty."""
        self._viewer = viewer

        # Per-frame allocation caches. None means "not built yet / invalidated".
        self._T_denorm_dev: Float[Tensor, "4 4"] | None = None
        self._bg_color: tuple[float, float, float] | None = None
        self._bg_tensor: Float[Tensor, "3"] | None = None
        # Static temporal colormap keyed by (model, colormap, mode); distance mode bypasses it.
        self._colormap_key: tuple[int, str, str] | None = None
        self._colors_override: Float[Tensor, "N 3"] | None = None
        self._validity_key: tuple | None = None
        self._validity_mask: Bool[Tensor, "1 H W 1"] | None = None

    def on_model_update(self) -> None:
        """Clear caches that depend on the model (the temporal colormap)."""
        self._colormap_key = None
        self._colors_override = None

    @staticmethod
    def viewmat_from_c2w(
        c2w: Float[Tensor, "4 4"],
        T_denorm: Float[Tensor, "4 4"] | None,
    ) -> Float[Tensor, "4 4"]:
        """World-to-camera matrix, moved into normalized space by ``T_denorm`` when present."""
        viewmat = torch.linalg.inv(c2w)
        if T_denorm is not None:
            viewmat = normalize_viewmat(viewmat, T_denorm)
        return viewmat

    @staticmethod
    def K_from_fov(
        fov_rad: float,
        width: int,
        height: int,
        device: torch.device,
    ) -> Float[Tensor, "3 3"]:
        """Pinhole intrinsics from a vertical field-of-view."""
        focal = height / 2.0 / np.tan(fov_rad / 2.0)
        return torch.tensor(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def fisheye_K(
        distortion: Float[Tensor, "4"],
        field_deg: float,
        circular: bool,
        width: int,
        height: int,
    ) -> Float[Tensor, "3 3"]:
        """Fisheye intrinsics whose image circle (the lens field, or the fold if nearer) fits the frame or its diagonal."""
        rim = fisheye_domain_radius(to_float64_numpy(distortion), math.radians(field_deg) / 2)
        radius = min(width, height) / 2 if circular else math.hypot(width, height) / 2
        focal = radius / rim
        return torch.tensor(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=distortion.device,
        )

    def _validity(self, camera: BatchedCamera) -> Bool[Tensor, "1 H W 1"] | None:
        """Pixels the lens observes (fold and fisheye field), cached per camera; ``None`` when all are."""
        if camera.distortion is None:
            return None
        field = self._viewer.render_settings.fisheye_field_deg / 2 if _is_fisheye(camera.model) else None
        K = camera.Ks[0].cpu()
        dist = camera.distortion[0].cpu()
        key = (camera.model, tuple(K.flatten().tolist()), tuple(dist.tolist()), camera.width, camera.height, field)
        if key != self._validity_key:
            mask = compute_validity(camera.model, K, dist, camera.width, camera.height, max_incidence_deg=field)
            self._validity_mask = None if bool(mask.all()) else mask.to(camera.device)[None, ..., None]
            self._validity_key = key
        return self._validity_mask

    def _t_denorm_dev(self) -> Float[Tensor, "4 4"] | None:
        """The normalised-space-to-world transform on device, moved once and reused."""
        if self._viewer._T_denorm is None:
            return None
        if self._T_denorm_dev is None:
            self._T_denorm_dev = self._viewer._T_denorm.to(self._viewer._device)
        return self._T_denorm_dev

    def _background_tensor(self) -> Float[Tensor, "3"]:
        """Background color tensor [1, 3], rebuilt only when the color changes."""
        color = self._viewer.render_settings.background_color
        if color != self._bg_color or self._bg_tensor is None:
            self._bg_tensor = torch.tensor([color], device=self._viewer._device, dtype=torch.float32)
            self._bg_color = color
        return self._bg_tensor

    def _temporal_colors_override(self, tau: float) -> Float[Tensor, "N 3"] | None:
        """Per-Gaussian colours for temporal render mode, else ``None``.

        ``mean`` colours by ``mu_t``, ``std`` by the marginal width ``sqrt(cov_4d[3, 3] + eps)`` (min-max
        normalized), both cached per (model, colormap, mode); ``distance`` colours by ``|tau - mu_t|`` every call.
        """
        viewer = self._viewer
        settings = viewer.render_settings
        if settings.render_mode != "temporal":
            return None

        model = viewer._model
        mode = settings.temporal_color_mode

        if mode == "distance":
            mu_t = model.splats["mu_t"]
            normalized = (tau - mu_t).abs().clamp(0, 1).unsqueeze(-1)
            return apply_float_colormap(normalized, settings.colormap)

        key = (id(model), settings.colormap, mode)
        if key != self._colormap_key or self._colors_override is None:
            if mode == "std":
                splats = model.splats
                cov_t_marg = temporal_marginal_from_rot4d(splats["quats_l"], splats["quats_r"], splats["scales_4d_log"])
                # The guarded width the renderer uses; an unguarded sqrt diverges on collapsed primitives.
                std_t_marg = temporal_marginal_std(cov_t_marg)
                lo = std_t_marg.min()
                normalized = ((std_t_marg - lo) / (std_t_marg.max() - lo + 1e-8)).unsqueeze(-1)
            else:  # mean
                normalized = (model.splats["mu_t"] + 0.5).clamp(0, 1).unsqueeze(-1)
            self._colors_override = apply_float_colormap(normalized, settings.colormap)
            self._colormap_key = key
        return self._colors_override

    def _render(
        self,
        c2w: Float[Tensor, "4 4"],
        K: Float[Tensor, "3 3"],
        width: int,
        height: int,
        tau: float,
        colors_override: Float[Tensor, "N 3"] | None,
    ) -> RenderOutput:
        """Camera math once → BatchedCamera → pipeline.render. Returns the raw output."""
        viewer = self._viewer
        viewmat = self.viewmat_from_c2w(c2w, self._t_denorm_dev())
        settings = viewer.render_settings
        model = settings.active_camera_model
        distortion = settings.active_distortion_params
        if distortion is None and model is not CameraModelType.PINHOLE:
            distortion = torch.zeros(distortion_width(model), device=K.device)
        if _is_fisheye(model):
            K = self.fisheye_K(distortion, settings.fisheye_field_deg, settings.fisheye_circular, width, height)
        # No observation mask: a free camera has no dataset entry, so the lens footprint is applied after the render.
        camera = BatchedCamera.from_single_view(viewmat, K, model, width, height, distortion)
        background = self._background_tensor()
        out = viewer._pipeline.render(
            viewer._model,
            camera,
            tau,
            viewer._model.sh_degree,
            backgrounds=background,
            colors_override=colors_override,
        )
        valid = self._validity(camera)
        if valid is None:
            return out
        rgb = torch.where(valid, out.colors[..., :3], background.view(1, 1, 1, 3))
        colors = torch.cat([rgb, out.colors[..., 3:] * valid], dim=-1)
        return replace(out, colors=colors, alphas=out.alphas * valid)

    @torch.no_grad()
    def render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState) -> UInt8[np.ndarray, "H W 3"]:
        """Interactive render callback for nerfview. Returns [H, W, 3] uint8."""
        viewer = self._viewer
        device = viewer._device
        tab = RenderTabView(render_tab_state)
        width = tab.viewer_width
        height = tab.viewer_height

        tau = tab.preview_time if tab.preview_render else viewer._temporal.tau

        c2w = torch.from_numpy(camera_state.c2w).float().to(device)

        viewer._apply_lens(device)

        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)

        colors_override = self._temporal_colors_override(tau)

        out = self._render(c2w, K, width, height, tau, colors_override)

        viewer._update_rendered_count(out.meta)
        image = viewer._postprocess(out)
        return (image * 255).astype(np.uint8)

    @torch.no_grad()
    def render_at(
        self,
        c2w: Float[np.ndarray, "4 4"],
        fov_rad: float,
        width: int,
        height: int,
        tau: float,
    ) -> UInt8[np.ndarray, "H W 3"]:
        """Offline export render ``[H, W, 3]`` uint8, through the same colour and post-processing path as the live view."""
        viewer = self._viewer
        device = viewer._device
        c2w_t = torch.from_numpy(c2w.astype(np.float32)).to(device)

        # Apply camera panels first, same as render_fn — otherwise this reads whatever camera model
        # was left over from the last interactive render, not the panel's current state.
        viewer._apply_lens(device)

        K = self.K_from_fov(fov_rad, width, height, device)

        colors_override = self._temporal_colors_override(tau)
        out = self._render(c2w_t, K, width, height, tau, colors_override)
        image = viewer._postprocess(out)
        return (image * 255).astype(np.uint8)
