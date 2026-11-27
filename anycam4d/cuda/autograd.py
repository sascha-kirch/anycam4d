"""Autograd wrappers for the CUDA conditioning, fused EWA, fused UT and world-rasterization kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import torch

from anycam4d.cuda.ext import launch_ewa_4d_fused_conditioned_projection_bwd_kernel
from anycam4d.cuda.ext import launch_ewa_4d_fused_conditioned_projection_fwd_kernel
from anycam4d.cuda.ext import launch_ewa_4d_fused_packed_conditioned_projection_bwd_kernel
from anycam4d.cuda.ext import launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel
from anycam4d.cuda.ext import rasterize_from_world_bwd
from anycam4d.cuda.ext import rasterize_from_world_fwd
from anycam4d.cuda.ext import ut_4d_fused_packed_projection_bwd
from anycam4d.cuda.ext import ut_4d_fused_packed_projection_fwd
from anycam4d.cuda.ext import ut_4d_fused_projection_bwd
from anycam4d.cuda.ext import ut_4d_fused_projection_fwd


if TYPE_CHECKING:
    from jaxtyping import Float
    from jaxtyping import Int
    from torch import Tensor


class RasterizeWorldFunction(torch.autograd.Function):
    """World-space ray-Gaussian rasterization (``D² = c - b²/a``) for both projection families.

    UT's differentiable compositing step; tile assignment happens separately without gradients.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        means_3d_cond: Float[Tensor, "N 3"],
        cov_factors: Float[Tensor, "N 6"],
        opacities: Float[Tensor, "N"],
        colors: Float[Tensor, "C N 3"],
        viewmats: Float[Tensor, "C 4 4"],
        Ks: Float[Tensor, "C 3 3"],
        radial_coeffs: Float[Tensor, "C P"],
        tile_offsets: Int[Tensor, "C tile_h tile_w"],
        flatten_ids: Int[Tensor, "n_isects"],
        image_width: int,
        image_height: int,
        family: int,
        valid_mask: Int[Tensor, "C H W_words"] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Composite front to back.

        ``radial_coeffs`` is fixed-width OpenCV distortion (fisheye reads four slots); ``family`` is 0 pinhole,
        1 fisheye; ``valid_mask`` is the bitpacked observation mask, ``None`` for fully observed.

        Returns:
            ``(render_colors [C, H, W, D], render_alphas [C, H, W])``.
        """
        render_colors, render_alphas, last_ids = rasterize_from_world_fwd(
            means_3d_cond,
            cov_factors,
            opacities,
            colors,
            viewmats,
            Ks,
            radial_coeffs,
            tile_offsets,
            flatten_ids,
            image_width,
            image_height,
            family,
            valid_mask,
        )
        ctx.save_for_backward(
            means_3d_cond,
            cov_factors,
            opacities,
            colors,
            viewmats,
            Ks,
            radial_coeffs,
            tile_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            valid_mask,
        )
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.family = family
        return render_colors, render_alphas

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        dL_drender_colors: Float[Tensor, "C H W CDIM"],
        dL_drender_alphas: Float[Tensor, "C H W 1"],
    ) -> tuple[Tensor | None, ...]:
        """Gradients for ``means_3d_cond``, ``cov_factors``, ``opacities`` and ``colors``; ``None`` for the rest."""
        (
            means_3d_cond,
            cov_factors,
            opacities,
            colors,
            viewmats,
            Ks,
            radial_coeffs,
            tile_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            valid_mask,
        ) = ctx.saved_tensors

        dL_dmeans_3d_cond, dL_dcov_factors, dL_dopacities, dL_dcolors = rasterize_from_world_bwd(
            means_3d_cond,
            cov_factors,
            opacities,
            colors,
            viewmats,
            Ks,
            radial_coeffs,
            tile_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            dL_drender_colors.contiguous(),
            dL_drender_alphas.contiguous(),
            ctx.family,
            valid_mask,
        )

        return (
            dL_dmeans_3d_cond,
            dL_dcov_factors,
            dL_dopacities,
            dL_dcolors,
            None,  # viewmats
            None,  # Ks
            None,  # radial_coeffs
            None,  # tile_offsets
            None,  # flatten_ids
            None,  # image_width
            None,  # image_height
            None,  # family
            None,  # valid_mask
        )


class EWAFusedPackedFunction(torch.autograd.Function):
    """Packed fused conditioning + SH + EWA projection over visible (camera, Gaussian) pairs.

    ``dist`` is ``[C, n_slots]`` distortion of projection ``family``, or empty for an undistorted pinhole.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        quats_l: Float[Tensor, "N 4"],
        quats_r: Float[Tensor, "N 4"],
        scales_4d_log: Float[Tensor, "N 4"],
        means: Float[Tensor, "N 3"],
        mu_t: Float[Tensor, "N"],
        opacities: Float[Tensor, "N"],
        sh0: Float[Tensor, "N 1 3"],
        shN: Float[Tensor, "N K 3"],
        sh_temporal: Float[Tensor, "N T 3"],
        sh_degree: int,
        appearance_kind: int,
        n_temporal_terms: int,
        temporal_stride: int,
        viewmats: Float[Tensor, "C 4 4"],
        Ks: Float[Tensor, "C 3 3"],
        dist: Float[Tensor, "C P"],
        family: int,
        tau: float,
        cov_eps: float,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        prefilter_threshold: float,
        image_width: int,
        image_height: int,
        calc_compensations: bool,
    ) -> tuple[Tensor, ...]:
        """Packed fused forward.

        Returns:
            ``(camera_ids [nnz], gaussian_ids [nnz], indptr [C+1], means2d [nnz, 2], conics [nnz, 3],
            colors [nnz, 3], opac [nnz], depths [nnz], radii [nnz, 2], means_3d_cond [N, 3], wt [N])``.
        """
        (
            camera_ids,
            gaussian_ids,
            indptr,
            means2d,
            conics,
            colors,
            opac,
            depths,
            radii,
            means_3d_cond,
            wt,
        ) = launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            viewmats,
            Ks,
            dist,
            family,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter_threshold,
            image_width,
            image_height,
            calc_compensations,
        )
        ctx.save_for_backward(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            viewmats,
            Ks,
            dist,
            camera_ids,
            gaussian_ids,
        )
        ctx.scalars = (
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter_threshold,
            image_width,
            image_height,
        )
        ctx.calc_compensations = calc_compensations
        ctx.family = family
        return camera_ids, gaussian_ids, indptr, means2d, conics, colors, opac, depths, radii, means_3d_cond, wt

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        _d_camera_ids: Float[Tensor, "nnz"],
        _d_gaussian_ids: Float[Tensor, "nnz"],
        _d_indptr: Float[Tensor, "C_plus_1"],
        d_means2d: Float[Tensor, "C N 2"],
        d_conics: Float[Tensor, "C N 3"],
        d_colors: Float[Tensor, "C N 3"],
        d_opac: Float[Tensor, "N"],
        _d_depths: Float[Tensor, "nnz"],
        _d_radii: Float[Tensor, "nnz 2"],
        _d_means3d: Float[Tensor, "N 3"],
        _d_wt: Float[Tensor, "N"],
    ) -> tuple[Tensor | None, ...]:
        """Backward pass over nnz visible pairs only."""
        (
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            viewmats,
            Ks,
            dist,
            camera_ids,
            gaussian_ids,
        ) = ctx.saved_tensors
        (
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter_threshold,
            image_width,
            image_height,
        ) = ctx.scalars

        (
            d_quats_l,
            d_quats_r,
            d_scales_4d_log,
            d_means,
            d_mu_t,
            d_opacities,
            d_sh0,
            d_shN,
            d_sh_temporal,
        ) = launch_ewa_4d_fused_packed_conditioned_projection_bwd_kernel(
            d_means2d.contiguous(),
            d_conics.contiguous(),
            d_colors.contiguous(),
            d_opac.contiguous(),
            camera_ids,
            gaussian_ids,
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            viewmats,
            Ks,
            dist,
            ctx.family,
            float(tau),
            float(cov_eps),
            float(eps2d),
            float(near_plane),
            float(far_plane),
            float(prefilter_threshold),
            int(image_width),
            int(image_height),
            ctx.calc_compensations,
        )

        return (
            d_quats_l,
            d_quats_r,
            d_scales_4d_log,
            d_means,
            d_mu_t,
            d_opacities,
            d_sh0,
            d_shN,
            d_sh_temporal if d_sh_temporal.numel() > 0 else None,
            None,  # sh_degree
            None,  # appearance_kind
            None,  # n_temporal_terms
            None,  # temporal_stride
            None,  # viewmats
            None,  # Ks
            None,  # dist
            None,  # family
            None,  # tau
            None,  # cov_eps
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # prefilter_threshold
            None,  # image_width
            None,  # image_height
            None,  # calc_compensations
        )


class EWAFusedFunction(torch.autograd.Function):
    """Fused conditioning + SH + EWA projection; only Gaussian parameters and SH receive gradients.

    ``dist`` is ``[C, n_slots]`` distortion of projection ``family``, or empty for an undistorted pinhole.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        quats_l: Float[Tensor, "N 4"],
        quats_r: Float[Tensor, "N 4"],
        scales_4d_log: Float[Tensor, "N 4"],
        means: Float[Tensor, "N 3"],
        mu_t: Float[Tensor, "N"],
        opacities: Float[Tensor, "N"],
        sh0: Float[Tensor, "N 1 3"],
        shN: Float[Tensor, "N K 3"],
        sh_temporal: Float[Tensor, "N T 3"],
        sh_degree: int,
        appearance_kind: int,
        n_temporal_terms: int,
        temporal_stride: int,
        viewmats: Float[Tensor, "C 4 4"],
        Ks: Float[Tensor, "C 3 3"],
        dist: Float[Tensor, "C P"],
        family: int,
        tau: float,
        cov_eps: float,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        prefilter_threshold: float,
        image_width: int,
        image_height: int,
        calc_compensations: bool,
    ) -> tuple[Tensor, ...]:
        """Fused forward; ``calc_compensations`` applies the Mip-Splatting opacity compensation.

        Returns:
            ``(means2d [C, N, 2], conics [C, N, 3], colors [C, N, 3], opac [C, N], depths [C, N],
            radii [C, N, 2], means_3d_cond [N, 3], wt [N])``.
        """
        means2d, conics, colors, opac, depths, radii, means_3d_cond, wt = (
            launch_ewa_4d_fused_conditioned_projection_fwd_kernel(
                quats_l,
                quats_r,
                scales_4d_log,
                means,
                mu_t,
                opacities,
                sh0,
                shN,
                sh_temporal,
                sh_degree,
                appearance_kind,
                n_temporal_terms,
                temporal_stride,
                viewmats,
                Ks,
                dist,
                family,
                tau,
                cov_eps,
                eps2d,
                near_plane,
                far_plane,
                prefilter_threshold,
                image_width,
                image_height,
                calc_compensations,
            )
        )
        # Save only the inputs needed for backward recomputation (no intermediates)
        ctx.save_for_backward(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            viewmats,
            Ks,
            dist,
        )
        ctx.scalars = (
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter_threshold,
            image_width,
            image_height,
        )
        ctx.calc_compensations = calc_compensations
        ctx.family = family
        return means2d, conics, colors, opac, depths, radii, means_3d_cond, wt

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        d_means2d: Float[Tensor, "C N 2"],
        d_conics: Float[Tensor, "C N 3"],
        d_colors: Float[Tensor, "C N 3"],
        d_opac: Float[Tensor, "N"],
        _d_depths: Float[Tensor, "C N"],  # not used (depths has no differentiable path here)
        _d_radii: Float[Tensor, "C N 2"],  # not used (radii are discrete)
        _d_means3d: Float[Tensor, "N 3"],  # not used (side output, stop-gradient)
        _d_wt: Float[Tensor, "N"],  # not used (side output, stop-gradient)
    ) -> tuple[Tensor | None, ...]:
        """Gradients for all forward inputs; depths, radii, ``means_3d_cond`` and ``wt`` are stop-gradient."""
        (
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            viewmats,
            Ks,
            dist,
        ) = ctx.saved_tensors
        (
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter_threshold,
            image_width,
            image_height,
        ) = ctx.scalars

        (
            d_quats_l,
            d_quats_r,
            d_scales_4d_log,
            d_means,
            d_mu_t,
            d_opacities,
            d_sh0,
            d_shN,
            d_sh_temporal,
        ) = launch_ewa_4d_fused_conditioned_projection_bwd_kernel(
            d_means2d.contiguous(),
            d_conics.contiguous(),
            d_colors.contiguous(),
            d_opac.contiguous(),
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            viewmats,
            Ks,
            dist,
            ctx.family,
            float(tau),
            float(cov_eps),
            float(eps2d),
            float(near_plane),
            float(far_plane),
            float(prefilter_threshold),
            int(image_width),
            int(image_height),
            ctx.calc_compensations,
        )

        # Return None for all non-differentiable args (ints, camera matrices, scalars)
        return (
            d_quats_l,
            d_quats_r,
            d_scales_4d_log,
            d_means,
            d_mu_t,
            d_opacities,
            d_sh0,
            d_shN,
            d_sh_temporal if d_sh_temporal.numel() > 0 else None,
            None,  # sh_degree
            None,  # appearance_kind
            None,  # n_temporal_terms
            None,  # temporal_stride
            None,  # viewmats
            None,  # Ks
            None,  # dist
            None,  # family
            None,  # tau
            None,  # cov_eps
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # prefilter_threshold
            None,  # image_width
            None,  # image_height
            None,  # calc_compensations
        )


class UTFusedFunction(torch.autograd.Function):
    """Fused conditioning + UT projection + 4D SH.

    Only ``means_3d_cond``, ``cov_factors``, ``opacities`` and ``colors`` carry gradients; the 2D projection
    outputs feed tile assignment only.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        quats_l: Float[Tensor, "N 4"],
        quats_r: Float[Tensor, "N 4"],
        scales_4d_log: Float[Tensor, "N 4"],
        means: Float[Tensor, "N 3"],
        mu_t: Float[Tensor, "N"],
        opacities: Float[Tensor, "N"],
        sh0: Float[Tensor, "N 1 3"],
        shN: Float[Tensor, "N K 3"],
        sh_temporal: Float[Tensor, "N T 3"],
        sh_degree: int,
        appearance_kind: int,
        n_temporal_terms: int,
        temporal_stride: int,
        viewmats: Float[Tensor, "C 4 4"],
        Ks: Float[Tensor, "C 3 3"],
        radial_coeffs: Float[Tensor, "C P"],
        tau: float,
        cov_eps: float,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        prefilter_threshold: float,
        image_width: int,
        image_height: int,
        family: int,
    ) -> tuple[Tensor, ...]:
        """Fused UT forward.

        Returns:
            ``(means2d [C, N, 2], conics [C, N, 3], depths [C, N], radii [C, N, 2], colors [C, N, 3],
            means_3d_cond [N, 3], cov_factors [N, 6], opacities [N], wt [N])``.
        """
        means2d, conics, depths, radii, colors, means_3d_cond, cov_factors, opac, wt = ut_4d_fused_projection_fwd(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            viewmats,
            Ks,
            radial_coeffs,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter_threshold,
            image_width,
            image_height,
            family,
        )
        ctx.save_for_backward(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            viewmats,
        )
        ctx.scalars = (
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            tau,
            cov_eps,
            prefilter_threshold,
        )
        return means2d, conics, depths, radii, colors, means_3d_cond, cov_factors, opac, wt

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        _d_means2d: Float[Tensor, "C N 2"],
        _d_conics: Float[Tensor, "C N 3"],
        _d_depths: Float[Tensor, "C N"],
        _d_radii: Float[Tensor, "C N 2"],
        d_colors: Float[Tensor, "C N 3"],
        d_means_3d_cond: Float[Tensor, "N 3"],
        d_cov_factors: Float[Tensor, "N 6"],
        d_opacities: Float[Tensor, "N"],
        _d_wt: Float[Tensor, "N"],
    ) -> tuple[Tensor | None, ...]:
        """Gradients for the eight differentiable inputs in forward order; ``None`` elsewhere."""
        (
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            viewmats,
        ) = ctx.saved_tensors
        (
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            tau,
            cov_eps,
            prefilter_threshold,
        ) = ctx.scalars

        (
            d_quats_l,
            d_quats_r,
            d_scales_4d_log,
            d_means,
            d_mu_t,
            d_opac_raw,
            d_sh0,
            d_shN,
            d_sh_temporal,
        ) = ut_4d_fused_projection_bwd(
            d_means_3d_cond.contiguous(),
            d_cov_factors.contiguous(),
            d_opacities.contiguous(),
            d_colors.contiguous(),
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            viewmats,
            float(tau),
            float(cov_eps),
            float(prefilter_threshold),
        )

        return (
            d_quats_l,
            d_quats_r,
            d_scales_4d_log,
            d_means,
            d_mu_t,
            d_opac_raw,
            d_sh0,
            d_shN,
            d_sh_temporal if d_sh_temporal.numel() > 0 else None,
            None,  # sh_degree
            None,  # appearance_kind
            None,  # n_temporal_terms
            None,  # temporal_stride
            None,  # viewmats
            None,  # Ks
            None,  # radial_coeffs
            None,  # tau
            None,  # cov_eps
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # prefilter_threshold
            None,  # image_width
            None,  # image_height
            None,  # family
        )


class UTFusedPackedFunction(torch.autograd.Function):
    """Packed UT path (count, then scatter).

    Tile-assignment outputs are packed; rasterization inputs stay dense, so the dense backward is reused.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        quats_l: Float[Tensor, "N 4"],
        quats_r: Float[Tensor, "N 4"],
        scales_4d_log: Float[Tensor, "N 4"],
        means: Float[Tensor, "N 3"],
        mu_t: Float[Tensor, "N"],
        opacities: Float[Tensor, "N"],
        sh0: Float[Tensor, "N 1 3"],
        shN: Float[Tensor, "N K 3"],
        sh_temporal: Float[Tensor, "N T 3"],
        sh_degree: int,
        appearance_kind: int,
        n_temporal_terms: int,
        temporal_stride: int,
        viewmats: Float[Tensor, "C 4 4"],
        Ks: Float[Tensor, "C 3 3"],
        radial_coeffs: Float[Tensor, "C P"],
        tau: float,
        cov_eps: float,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        prefilter_threshold: float,
        image_width: int,
        image_height: int,
        family: int,
    ) -> tuple[Tensor, ...]:
        """Packed UT forward.

        Returns:
            ``(camera_ids, gaussian_ids, indptr, means2d, conics, depths, radii)`` packed over ``nnz``, then
            dense ``colors [C, N, 3], means_3d_cond [N, 3], cov_factors [N, 6], opacities [N], wt [N]``.
        """
        (
            camera_ids,
            gaussian_ids,
            indptr,
            means2d,
            conics,
            depths,
            radii,
            colors,
            means_3d_cond,
            cov_factors,
            opac,
            wt,
        ) = ut_4d_fused_packed_projection_fwd(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            viewmats,
            Ks,
            radial_coeffs,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter_threshold,
            image_width,
            image_height,
            family,
        )
        ctx.save_for_backward(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            viewmats,
            camera_ids,
            gaussian_ids,
        )
        ctx.scalars = (
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            tau,
            cov_eps,
            prefilter_threshold,
        )
        return (
            camera_ids,
            gaussian_ids,
            indptr,
            means2d,
            conics,
            depths,
            radii,
            colors,
            means_3d_cond,
            cov_factors,
            opac,
            wt,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        _d_camera_ids: Float[Tensor, "nnz"],
        _d_gaussian_ids: Float[Tensor, "nnz"],
        _d_indptr: Float[Tensor, "C_plus_1"],
        _d_means2d: Float[Tensor, "nnz 2"],
        _d_conics: Float[Tensor, "nnz 3"],
        _d_depths: Float[Tensor, "nnz"],
        _d_radii: Float[Tensor, "nnz 2"],
        d_colors: Float[Tensor, "C N 3"],
        d_means_3d_cond: Float[Tensor, "N 3"],
        d_cov_factors: Float[Tensor, "N 6"],
        d_opacities: Float[Tensor, "N"],
        _d_wt: Float[Tensor, "N"],
    ) -> tuple[Tensor | None, ...]:
        """Backward via ut_4d_fused_packed_projection_bwd — parallelizes over nnz visible pairs."""
        (
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            viewmats,
            camera_ids,
            gaussian_ids,
        ) = ctx.saved_tensors
        (
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            tau,
            cov_eps,
            prefilter_threshold,
        ) = ctx.scalars

        (
            d_quats_l,
            d_quats_r,
            d_scales_4d_log,
            d_means,
            d_mu_t,
            d_opac_raw,
            d_sh0,
            d_shN,
            d_sh_temporal,
        ) = ut_4d_fused_packed_projection_bwd(
            d_colors.contiguous(),
            d_means_3d_cond.contiguous(),
            d_cov_factors.contiguous(),
            d_opacities.contiguous(),
            camera_ids,
            gaussian_ids,
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            appearance_kind,
            n_temporal_terms,
            temporal_stride,
            viewmats,
            float(tau),
            float(cov_eps),
            float(prefilter_threshold),
        )

        return (
            d_quats_l,
            d_quats_r,
            d_scales_4d_log,
            d_means,
            d_mu_t,
            d_opac_raw,
            d_sh0,
            d_shN,
            d_sh_temporal if d_sh_temporal.numel() > 0 else None,
            None,  # sh_degree
            None,  # appearance_kind
            None,  # n_temporal_terms
            None,  # temporal_stride
            None,  # viewmats
            None,  # Ks
            None,  # radial_coeffs
            None,  # tau
            None,  # cov_eps
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # prefilter_threshold
            None,  # image_width
            None,  # image_height
            None,  # family
        )
