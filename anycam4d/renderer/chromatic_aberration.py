"""Lateral chromatic aberration in one rasterizer pass, through any lens the UT supports.

Each colour channel integrates ``samples`` wavelengths across its band; wavelength λ sees the camera with its focal
length scaled by the magnification m(λ), a Cauchy dispersion curve with m = 1 at green and m(red) - m(blue) =
``strength``. Every pixel casts one ray per wavelength from the same centre, so conditioning, colour, depth order and
tile lists are shared; the tile box is the union of the per-wavelength boxes. Unobserved pixels (``valid_mask``) stay
empty, as in the plain render. RGB radiance stands in for each band's spectrum. Forward only.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import torch
from gsplat.cuda._wrapper import isect_offset_encode
from gsplat.cuda._wrapper import isect_tiles

from anycam4d.cuda.camera import to_kernel_args
from anycam4d.cuda.ext import UT_WORLD_RASTERIZE_TILE_SIZE as _TILE_SIZE
from anycam4d.cuda.ext import rasterize_from_world_dispersion_fwd
from anycam4d.cuda.ext import ut_4d_dispersion_projection_fwd
from anycam4d.enums import BackendType
from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE
from anycam4d.renderer.capability import check_camera_capability
from anycam4d.renderer.render_registry import check_render_capability


if TYPE_CHECKING:
    from jaxtyping import Float
    from jaxtyping import Int
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.cuda.camera import KernelCameraArgs
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

BANDS_NM = ((580.0, 680.0), (490.0, 580.0), (420.0, 490.0))  # R, G, B sensor bands
SAMPLES = (1, 2, 4, 8)  # wavelengths per channel the rasterizer is compiled for
RED_NM, GREEN_NM, BLUE_NM = 630.0, 535.0, 455.0  # band centres


def wavelengths(samples: int) -> list[float]:
    """``samples`` stratified wavelengths per band, red band first; one sample is the band centre."""
    return [lo + (i + 0.5) / samples * (hi - lo) for lo, hi in BANDS_NM for i in range(samples)]


def magnification(wavelength: float, strength: float) -> float:
    """Cauchy dispersion: 1 at green, m(red) - m(blue) = ``strength``."""
    span = BLUE_NM**-2 - RED_NM**-2
    return 1.0 + strength * (GREEN_NM**-2 - wavelength**-2) / span


def dispersion_lenses(camera: BatchedCamera, strength: float, samples: int) -> BatchedCamera:
    """``camera`` repeated once per wavelength, its focal length scaled by m(λ) about the principal point."""
    scale = torch.tensor([magnification(w, strength) for w in wavelengths(samples)], device=camera.Ks.device)
    Ks = camera.Ks.expand(len(scale), 3, 3).clone()
    Ks[:, 0, 0] *= scale
    Ks[:, 1, 1] *= scale
    return dataclasses.replace(
        camera,
        viewmats=camera.viewmats.expand(len(scale), 4, 4).contiguous(),
        Ks=Ks,
        distortion=None if camera.distortion is None else camera.distortion.expand(len(scale), -1).contiguous(),
        valid_mask=None,  # one sensor: the caller passes camera.valid_mask to the rasterizer once
    )


def render_chromatic_aberration(
    model: FourDGaussians,
    camera: BatchedCamera,
    tau: float,
    sh_degree: int,
    cfg: RenderingConfig,
    *,
    strength: float,
    samples: int = 1,
    sh_degree_t: int = MAX_TEMPORAL_DEGREE,
) -> tuple[Float[Tensor, "1 H W 3"], Float[Tensor, "1 H W 3"]]:
    """One pass with ``samples`` wavelengths per channel (1, 2, 4 or 8); returns colours and per-channel alphas."""
    if cfg.backend != BackendType.ANYCAM4D:
        raise ValueError(
            f"chromatic aberration renders through the UT kernels; rendering.backend is {cfg.backend.value}"
        )
    if samples not in SAMPLES:
        raise ValueError(f"samples must be one of {SAMPLES}, got {samples}")
    if camera.viewmats.shape[0] != 1:
        raise ValueError(f"chromatic aberration renders one camera, got {camera.viewmats.shape[0]}")
    check_render_capability(BackendType.ANYCAM4D, cfg.rasterize_mode)
    check_camera_capability(camera.model, BackendType.ANYCAM4D, camera.distortion)

    args = to_kernel_args(dispersion_lenses(camera, strength, samples))
    with torch.no_grad():
        centre, depth, half, colors, means, cov_factors, opacities = project(
            model, args, tau, sh_degree, cfg, sh_degree_t
        )
        tile_width = math.ceil(args.width / _TILE_SIZE)
        tile_height = math.ceil(args.height / _TILE_SIZE)
        _, isect_ids, flatten_ids = isect_tiles(
            centre, half, depth, _TILE_SIZE, tile_width, tile_height, packed=False, n_images=1
        )
        offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height).reshape(1, tile_height, tile_width)
        return rasterize_from_world_dispersion_fwd(
            means,
            cov_factors,
            opacities,
            colors,
            args.viewmats[:1].contiguous(),
            args.Ks,
            args.distortion,
            offsets,
            flatten_ids,
            args.width,
            args.height,
            args.family,
            camera.valid_mask,
        )


def project(
    model: FourDGaussians, args: KernelCameraArgs, tau: float, sh_degree: int, cfg: RenderingConfig, sh_degree_t: int
) -> tuple[
    Float[Tensor, "1 N 2"],
    Float[Tensor, "1 N"],
    Int[Tensor, "1 N 2"],
    Float[Tensor, "1 N 3"],
    Float[Tensor, "N 3"],
    Float[Tensor, "N 6"],
    Float[Tensor, "N"],
]:
    """Condition and colour once; the tile box is the union of the lenses' UT boxes.

    Returns centre, depth, half-size, colour, and the conditioned means, covariance factors and opacities.
    """
    splats = model.splats
    appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)
    return ut_4d_dispersion_projection_fwd(
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
        args.viewmats[:1].contiguous(),
        args.Ks,
        args.distortion,
        float(tau),
        model.cov_eps,
        cfg.eps2d,
        cfg.near_plane,
        cfg.far_plane,
        model.prefilter_threshold,
        args.width,
        args.height,
        args.family,
    )
