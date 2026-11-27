"""Reference UT backend in three stages.

1. Tile assignment (no grad): ``tests.fused_stages.ut_project`` and gsplat's tile intersection.
2. Conditioning: ``tests.fused_stages.condition``.
3. Compositing: ``RasterizeWorldFunction`` against the conditioned 3D covariance.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from gsplat.cuda._wrapper import isect_offset_encode
from gsplat.cuda._wrapper import isect_tiles

from tests.fused_stages import condition
from anycam4d.cuda.autograd import RasterizeWorldFunction
from anycam4d.renderer.backend import RenderOutput
from tests.oracles.ewa_pytorch_backend import EWAPyTorchBackend


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from tests.oracles.sh4d import Sh4dAppearanceModel as AppearanceModelProto
    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.cuda.ext import BUILD_ERROR
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import UT_WORLD_RASTERIZE_TILE_SIZE as _TILE_SIZE
from anycam4d.cuda.camera import to_kernel_args
from anycam4d.enums import BackendType


if not HAS_CUDA_EXT:
    raise ImportError(
        "UTBackend requires the anycam4d CUDA extension, which failed to build; `pixi run cuda-build` shows why."
    ) from BUILD_ERROR


class UTPyTorchBackend:
    """Three-stage, non-fused UT reference; UT constants are compiled into the kernels."""

    def __init__(
        self,
        cfg: RenderingConfig,
        appearance_model: AppearanceModelProto | None = None,
    ) -> None:
        """Initialise UTBackend."""
        self._cfg = cfg
        self._appearance_model = appearance_model
        self._pytorch_fallback = EWAPyTorchBackend(cfg, appearance_model)

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
        backgrounds: Float[Tensor, "C 3"] | None = None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Render a camera batch through the three-stage UT path."""
        assert BackendType.ANYCAM4D in CAMERA_REGISTRY[camera.model].backends, (  # noqa: S101
            f"UTPyTorchBackend does not support {camera.model}"
        )
        if colors_override is not None:
            return self._pytorch_fallback.render(
                model,
                camera,
                tau,
                sh_degree,
                backgrounds=backgrounds,
                colors_override=colors_override,
            )
        # Carry the camera into the render helper; it converts to kernel args at the boundary.
        return self._render_ut(model, camera, tau, sh_degree, backgrounds)

    def _render_ut(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        tau: float,
        sh_degree: int,
        backgrounds: Float[Tensor, "C 3"] | None,
    ) -> RenderOutput:
        """Core UT render: tile assignment (no_grad) + world rasterization (grad)."""
        from tests.fused_stages import fused_ut_project

        splats = model.splats
        # Convert + unpack at the kernel boundary (launch_*_kernel / RasterizeWorldFunction.apply).
        args = to_kernel_args(camera)
        viewmats = args.viewmats
        Ks = args.Ks
        rc = args.distortion
        family = args.family
        width = args.width
        height = args.height
        C = viewmats.shape[0]

        with torch.no_grad():
            radii, means2d, depths, conics = fused_ut_project(
                splats["quats_l"],
                splats["quats_r"],
                splats["scales_4d_log"],
                splats["means"],
                splats["mu_t"],
                splats["opacities"],
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

        means_3d_cond, covars, opacities, wt = condition(
            splats["quats_l"],
            splats["quats_r"],
            splats["scales_4d_log"],
            splats["means"],
            splats["mu_t"],
            splats["opacities"],
            float(tau),
            model.cov_eps,
            model.prefilter_threshold,
        )

        if self._appearance_model is not None:
            colors_per_cam = []
            for c in range(C):
                rgb = self._appearance_model.eval_colors(
                    splats, means_3d_cond, viewmats[c : c + 1], float(tau), sh_degree
                )
                colors_per_cam.append(rgb)
            colors_cn = torch.stack(colors_per_cam, dim=0)
        else:
            K_current = (sh_degree + 1) ** 2
            sh_coeffs = torch.cat([splats["sh0"], splats["shN"]], dim=1)[:, :K_current, :]
            colors_cn = self._eval_sh_per_camera(means_3d_cond, sh_coeffs, viewmats, sh_degree, C)

        render_colors, render_alphas = RasterizeWorldFunction.apply(
            means_3d_cond,
            covars,
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
        )

        if backgrounds is not None:
            bg = backgrounds[:, None, None, :]
            render_colors = render_colors + bg * (1.0 - render_alphas.unsqueeze(-1))

        meta = {
            "radii": radii,
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

    @staticmethod
    def _eval_sh_per_camera(
        means_3d_cond: Tensor,
        sh_coeffs: Tensor,
        viewmats: Tensor,
        sh_degree: int,
        C: int,
    ) -> Tensor:
        """Evaluate SH colours for each (camera, Gaussian) pair."""
        from gsplat import spherical_harmonics

        N = means_3d_cond.shape[0]
        R = viewmats[:, :3, :3]
        t = viewmats[:, :3, 3]
        cam_pos = -torch.einsum("cij,ci->cj", R, t)

        dirs = means_3d_cond[None, :, :] - cam_pos[:, None, :]
        dirs_flat = dirs.reshape(C * N, 3)

        sh_coeffs_flat = (
            sh_coeffs[None, :, :, :].expand(C, N, *sh_coeffs.shape[1:]).reshape(C * N, *sh_coeffs.shape[1:])
        )
        rgb = spherical_harmonics(sh_degree, dirs_flat, sh_coeffs_flat)
        rgb = torch.clamp_min(rgb + 0.5, 0.0)
        return rgb.reshape(C, N, 3)

    def condition_means(
        self,
        model: FourDGaussians,
        tau: float,
    ) -> Float[Tensor, "N 3"]:
        """Return conditioned 3D means only (lightweight, delegates to fallback)."""
        return self._pytorch_fallback.condition_means(model, tau)
