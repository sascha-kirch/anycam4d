"""Fused UT backend: conditioning, UT projection and 4D SH in one kernel, then world-space rasterization.

Dense: ``FusedUTFunction`` -> ``isect_tiles`` -> ``RasterizeWorldFunction``. Packed:
``FusedUTPackedFunction`` -> packed ``isect_tiles`` -> ``RasterizeWorldFunction`` on dense inputs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from gsplat.cuda._wrapper import isect_offset_encode
from gsplat.cuda._wrapper import isect_tiles

from anycam4d.cuda.autograd import RasterizeWorldFunction
from anycam4d.cuda.autograd import UTFusedFunction
from anycam4d.cuda.autograd import UTFusedPackedFunction
from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE
from anycam4d.renderer.backend import RenderOutput


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

from anycam4d.cuda.camera import to_kernel_args
from anycam4d.cuda.ext import BUILD_ERROR
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import UT_WORLD_RASTERIZE_TILE_SIZE as _TILE_SIZE


if not HAS_CUDA_EXT:
    raise ImportError(
        "FusedUTBackend requires the anycam4d CUDA extension, which failed to build; `pixi run cuda-build` shows why."
    ) from BUILD_ERROR


class UTFusedBackend:
    """Fused UT render backend, dense or packed per ``cfg.packed``; UT constants are compiled into the kernels."""

    def __init__(
        self,
        cfg: RenderingConfig,
    ) -> None:
        """Initialise FusedUTBackend."""
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
        """Render any registered camera model; ``colors_override`` replaces the kernel's SH colour."""
        # Carry the camera into the render helpers; they convert to kernel args at the boundary.
        if self._cfg.packed:
            return self._render_ut_fused_packed(
                model, camera, tau, sh_degree, sh_degree_t, backgrounds, colors_override
            )
        return self._render_ut_fused(model, camera, tau, sh_degree, sh_degree_t, backgrounds, colors_override)

    def _render_ut_fused(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        tau: float,
        sh_degree: int,
        sh_degree_t: int,
        backgrounds: Float[Tensor, "C 3"] | None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Dense fused UT render: FusedUTFunction → tile assignment → world rasterize."""
        splats = model.splats
        # Convert + unpack at the autograd boundary (immediately before UTFusedFunction.apply).
        args = to_kernel_args(camera)
        viewmats = args.viewmats
        Ks = args.Ks
        rc = args.distortion
        family = args.family
        width = args.width
        height = args.height
        C = viewmats.shape[0]

        appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)

        means2d, conics, depths, radii, colors_cn, means_3d_cond, cov_factors, opacities, _wt = UTFusedFunction.apply(
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
            rc,
            float(tau),
            model.cov_eps,
            self._cfg.eps2d,
            self._cfg.near_plane,
            self._cfg.far_plane,
            model.prefilter_threshold,
            width,
            height,
            family,
        )

        tile_width = math.ceil(width / _TILE_SIZE)
        tile_height = math.ceil(height / _TILE_SIZE)

        with torch.no_grad():
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

        # Color override (viewer) swaps the fused-kernel color before the world rasterizer;
        # the kernel's SH color was already computed but is simply discarded in this mode.
        if colors_override is not None:
            colors_cn = colors_override.unsqueeze(0).expand(C, -1, -1).contiguous()

        render_colors, render_alphas = RasterizeWorldFunction.apply(
            means_3d_cond,
            cov_factors,
            opacities,
            colors_cn,
            viewmats,
            Ks,
            rc,
            isect_offsets,
            flatten_ids,
            width,
            height,
            family,
            args.valid_mask,
        )

        if backgrounds is not None:
            bg = backgrounds[:, None, None, :]
            render_colors = render_colors + bg * (1.0 - render_alphas.unsqueeze(-1))

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
            "packed": False,
            "width": width,
            "height": height,
            "tile_size": _TILE_SIZE,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "isect_offsets": isect_offsets,
            "flatten_ids": flatten_ids,
        }

        return RenderOutput(colors=render_colors, alphas=render_alphas.unsqueeze(-1), meta=meta)

    def _render_ut_fused_packed(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        tau: float,
        sh_degree: int,
        sh_degree_t: int,
        backgrounds: Float[Tensor, "C 3"] | None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Packed tile assignment over dense rasterization inputs; packed ids are remapped to dense ones."""
        splats = model.splats
        # Convert + unpack at the autograd boundary (immediately before UTFusedPackedFunction.apply).
        args = to_kernel_args(camera)
        viewmats = args.viewmats
        Ks = args.Ks
        rc = args.distortion
        family = args.family
        width = args.width
        height = args.height
        C = viewmats.shape[0]

        appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)

        (
            camera_ids,
            gaussian_ids,
            _indptr,
            means2d,
            conics,
            depths,
            radii,
            colors_cn,
            means_3d_cond,
            cov_factors,
            opacities,
            _wt,
        ) = UTFusedPackedFunction.apply(
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
            rc,
            float(tau),
            model.cov_eps,
            self._cfg.eps2d,
            self._cfg.near_plane,
            self._cfg.far_plane,
            model.prefilter_threshold,
            width,
            height,
            family,
        )

        tile_width = math.ceil(width / _TILE_SIZE)
        tile_height = math.ceil(height / _TILE_SIZE)

        _, isect_ids, flatten_ids_packed = isect_tiles(
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

        # Packed ids -> dense cam * N + gauss: the rasterizer indexes geometry by id % N and colours by the full id.
        # Dropping cam * N would give every camera camera 0's colours.
        n_gauss = means_3d_cond.shape[0]
        flatten_ids = (camera_ids[flatten_ids_packed] * n_gauss + gaussian_ids[flatten_ids_packed]).to(torch.int32)

        # Color override (viewer) swaps the fused-kernel color before the world rasterizer;
        # the packed kernel's colors are already dense [C, N, 3], so the swap is a plain replace.
        if colors_override is not None:
            colors_cn = colors_override.unsqueeze(0).expand(C, -1, -1).contiguous()

        render_colors, render_alphas = RasterizeWorldFunction.apply(
            means_3d_cond,
            cov_factors,
            opacities,
            colors_cn,
            viewmats,
            Ks,
            rc,
            isect_offsets,
            flatten_ids,
            width,
            height,
            family,
            args.valid_mask,
        )

        if backgrounds is not None:
            bg = backgrounds[:, None, None, :]
            render_colors = render_colors + bg * (1.0 - render_alphas.unsqueeze(-1))

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

        return RenderOutput(colors=render_colors, alphas=render_alphas.unsqueeze(-1), meta=meta)
