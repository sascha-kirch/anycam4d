"""Fused EWA backend: conditioning, SH and EWA projection in one kernel, then gsplat compositing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from gsplat.cuda._wrapper import isect_offset_encode
from gsplat.cuda._wrapper import isect_tiles
from gsplat.cuda._wrapper import rasterize_to_pixels

from anycam4d.cuda.autograd import EWAFusedFunction
from anycam4d.cuda.autograd import EWAFusedPackedFunction
from anycam4d.cuda.camera import to_kernel_args
from anycam4d.enums import CameraModelType
from anycam4d.enums import RasterizeMode
from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE
from anycam4d.renderer.backend import RenderOutput


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

_TILE_SIZE = 16


def _lens_distortion(camera: BatchedCamera) -> tuple[Tensor, int]:
    """Kernel ``(dist, family)``: empty on an undistorted PINHOLE, which keeps the pinhole device path."""
    if camera.model == CameraModelType.PINHOLE:
        return torch.empty((0,), device=camera.viewmats.device), 0
    args = to_kernel_args(camera)
    return args.distortion, args.family


class FusedEWABackend:
    """Fused EWA render backend for training and inference; covariance through the lens's local Jacobian.

    With ``colors_override`` the kernel still provides geometry and its SH colour is replaced.
    """

    def __init__(
        self,
        cfg: RenderingConfig,
    ) -> None:
        """Initialize FusedEWABackend."""
        self._cfg = cfg

    @property
    def cfg(self) -> RenderingConfig:
        """Rendering configuration."""
        return self._cfg

    def render(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        tau: float,
        sh_degree: int,
        *,
        sh_degree_t: int = MAX_TEMPORAL_DEGREE,
        backgrounds: Float[Tensor, "C 3"] | None = None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Render a camera batch through its own lens."""
        # Carry the camera into the render helpers; they unpack at the autograd boundary.
        if self._cfg.packed:
            return self._render_fused_packed(model, camera, tau, sh_degree, sh_degree_t, backgrounds, colors_override)
        return self._render_fused(model, camera, tau, sh_degree, sh_degree_t, backgrounds, colors_override)

    def _render_fused(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        tau: float,
        sh_degree: int,
        sh_degree_t: int,
        backgrounds: Float[Tensor, "C 3"] | None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Fused kernel → gsplat tiling → gsplat compositing (training + inference)."""
        splats = model.splats
        # Unpack at the autograd boundary (immediately before EWAFusedFunction.apply).
        viewmats = camera.viewmats
        Ks = camera.Ks
        dist, family = _lens_distortion(camera)
        width = camera.width
        height = camera.height
        C = viewmats.shape[0]

        appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)

        # Launch fused kernel (differentiable via FusedEWAFunction)
        calc_compensations = self._cfg.rasterize_mode == RasterizeMode.ANTIALIASED
        means2d, conics, colors, opacities, depths, radii, means_3d_cond, _wt = EWAFusedFunction.apply(
            splats["quats_l"],
            splats["quats_r"],
            splats["scales_4d_log"],
            splats["means"],
            splats["mu_t"],
            splats["opacities"],
            appearance.sh0,
            appearance.shN,
            appearance.coeffs,
            sh_degree,
            appearance.kind_id,
            appearance.n_terms,
            appearance.stride,
            viewmats,
            Ks,
            dist,
            family,
            float(tau),
            model.cov_eps,
            self._cfg.eps2d,
            self._cfg.near_plane,
            self._cfg.far_plane,
            model.prefilter_threshold,
            width,
            height,
            calc_compensations,
        )

        # Tiling: map Gaussians to intersecting tiles
        tile_width = math.ceil(width / _TILE_SIZE)
        tile_height = math.ceil(height / _TILE_SIZE)

        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            _TILE_SIZE,
            tile_width,
            tile_height,
            packed=False,
            n_images=C,
        )
        isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
        isect_offsets = isect_offsets.reshape(C, tile_height, tile_width)

        # Color override (viewer) swaps the fused-kernel color before the compositor; the
        # kernel's SH color was already computed but is simply discarded in this mode.
        colors_cn = (
            colors_override.unsqueeze(0).expand(C, -1, -1).contiguous() if colors_override is not None else colors
        )

        # Compositing
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors_cn,
            opacities,
            width,
            height,
            _TILE_SIZE,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=False,
        )

        # Alpha-blend background if provided (same logic as PyTorchEWABackend)
        if backgrounds is not None:
            bg = backgrounds[:, None, None, :]  # [C, 1, 1, 3]
            render_colors[..., :3] = render_colors[..., :3] + bg * (1.0 - render_alphas)

        meta = {
            "radii": radii,
            "means_3d_cond": means_3d_cond.detach(),
            "viewmats": viewmats,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
            "gaussian_ids": None,
            "camera_ids": None,
            "n_cameras": C,
            # The fused renderer always runs isect_tiles/rasterize_to_pixels in non-packed
            # mode regardless of the rendering config.  Record the actual mode so the
            # trainer passes the correct value to the densification strategy.
            "packed": False,
            "width": width,
            "height": height,
            "tile_size": _TILE_SIZE,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "isect_offsets": isect_offsets,
            "flatten_ids": flatten_ids,
        }

        return RenderOutput(colors=render_colors, alphas=render_alphas, meta=meta)

    def _render_fused_packed(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        tau: float,
        sh_degree: int,
        sh_degree_t: int,
        backgrounds: Float[Tensor, "C 3"] | None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Packed fused kernel → gsplat tiling → gsplat compositing."""
        splats = model.splats
        # Unpack at the autograd boundary (immediately before EWAFusedPackedFunction.apply).
        viewmats = camera.viewmats
        Ks = camera.Ks
        dist, family = _lens_distortion(camera)
        width = camera.width
        height = camera.height
        C = viewmats.shape[0]

        appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)

        calc_compensations = self._cfg.rasterize_mode == RasterizeMode.ANTIALIASED
        (
            camera_ids,
            gaussian_ids,
            _indptr,
            means2d,
            conics,
            colors,
            opacities,
            depths,
            radii,
            means_3d_cond,
            _wt,
        ) = EWAFusedPackedFunction.apply(
            splats["quats_l"],
            splats["quats_r"],
            splats["scales_4d_log"],
            splats["means"],
            splats["mu_t"],
            splats["opacities"],
            appearance.sh0,
            appearance.shN,
            appearance.coeffs,
            sh_degree,
            appearance.kind_id,
            appearance.n_terms,
            appearance.stride,
            viewmats,
            Ks,
            dist,
            family,
            float(tau),
            model.cov_eps,
            self._cfg.eps2d,
            self._cfg.near_plane,
            self._cfg.far_plane,
            model.prefilter_threshold,
            width,
            height,
            calc_compensations,
        )

        # Tiling: packed mode
        tile_width = math.ceil(width / _TILE_SIZE)
        tile_height = math.ceil(height / _TILE_SIZE)

        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            _TILE_SIZE,
            tile_width,
            tile_height,
            packed=True,
            n_images=C,
            image_ids=camera_ids,
            gaussian_ids=gaussian_ids,
        )
        isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
        isect_offsets = isect_offsets.reshape(C, tile_height, tile_width)

        # Color override (viewer) swaps the fused-kernel color before the compositor.
        # ``colors`` is packed [nnz, 3] (one entry per visible camera/Gaussian pair); gather
        # the same pairs out of the dense [N, 3] override via gaussian_ids.
        colors_cn = colors_override[gaussian_ids] if colors_override is not None else colors

        # Packed compositing takes a (channels,) background, so backgrounds are blended manually afterwards.
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors_cn,
            opacities,
            width,
            height,
            _TILE_SIZE,
            isect_offsets,
            flatten_ids,
            packed=True,
        )

        if backgrounds is not None:
            bg = backgrounds[:, None, None, :]  # [C, 1, 1, 3]
            render_colors[..., :3] = render_colors[..., :3] + bg * (1.0 - render_alphas)

        meta = {
            "radii": radii,
            "means_3d_cond": means_3d_cond.detach(),
            "viewmats": viewmats,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
            "gaussian_ids": gaussian_ids,
            "camera_ids": camera_ids,
            "n_cameras": C,
            "packed": True,
            "width": width,
            "height": height,
            "tile_size": _TILE_SIZE,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "isect_offsets": isect_offsets,
            "flatten_ids": flatten_ids,
        }

        return RenderOutput(colors=render_colors, alphas=render_alphas, meta=meta)
