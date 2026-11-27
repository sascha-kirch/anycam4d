"""Thin-lens depth of field through the UT kernels, for one camera and any lens the UT supports.

Aperture disk of radius ``aperture`` in the camera's xy plane, focus plane at camera depth ``focus`` (scene units).
One pass of the kernel renders the aperture point ``offset`` exactly (the scene sheared so the focus plane stays fixed)
and blurs every Gaussian by a disk of ``blur_radius`` at its depth. ``samples`` passes at Vogel points of the aperture,
each blurred by the residual radius, trade cost for accuracy: one blurred pass composites blurred layers in depth order
rather than averaging them, which shows on high-contrast edges. The shared 1/255 alpha cutoff drops a strongly
blurred Gaussian's faint tails: covered by neighbours in a dense scene, dimmer over empty background. Forward only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from gsplat.cuda._wrapper import isect_offset_encode
from gsplat.cuda._wrapper import isect_tiles

from anycam4d.cuda.camera import to_kernel_args
from anycam4d.cuda.ext import UT_WORLD_RASTERIZE_TILE_SIZE as _TILE_SIZE
from anycam4d.cuda.ext import rasterize_from_world_fwd
from anycam4d.cuda.ext import ut_4d_depth_of_field_projection_fwd
from anycam4d.enums import BackendType
from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE
from anycam4d.renderer.capability import check_camera_capability
from anycam4d.renderer.render_registry import check_render_capability


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def vogel_disk(radius: float, n: int) -> list[tuple[float, float]]:
    """``n`` points evenly covering a disk of ``radius`` (golden-angle spiral)."""
    points = []
    for i in range(n):
        r = radius * math.sqrt((i + 0.5) / n)
        points.append((r * math.cos(i * GOLDEN_ANGLE), r * math.sin(i * GOLDEN_ANGLE)))
    return points


def pass_offsets(aperture: float, samples: int) -> list[tuple[float, float]]:
    """Vogel offsets, centred and given covariance ``aperture²/4 · (1 - 1/samples) · I``.

    With each pass blurred by a disk of ``aperture / √samples`` (covariance ``aperture²/(4 samples) · I``) the passes
    sum to the full aperture's second moment; a few raw Vogel points are neither centred nor isotropic. Two points are
    collinear and cannot be made isotropic, so ``samples`` is 1 or at least 3.
    """
    if samples < 1 or samples == 2:
        raise ValueError(f"samples must be 1 or >= 3, got {samples}")
    if samples == 1 or aperture == 0.0:
        return [(0.0, 0.0)]
    pts = np.asarray(vogel_disk(aperture, samples))
    pts -= pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(pts.T @ pts / samples)
    whiten = evecs @ np.diag(evals**-0.5) @ evecs.T
    pts = pts @ whiten * (aperture / 2.0 * math.sqrt(1.0 - 1.0 / samples))
    return [(float(x), float(y)) for x, y in pts]


def render_depth_of_field(
    model: FourDGaussians,
    camera: BatchedCamera,
    tau: float,
    sh_degree: int,
    cfg: RenderingConfig,
    *,
    focus: float,
    aperture: float,
    samples: int = 1,
    sh_degree_t: int = MAX_TEMPORAL_DEGREE,
) -> tuple[Float[Tensor, "1 H W 3"], Float[Tensor, "1 H W 1"]]:
    """Mean of ``samples`` thin-lens passes at ``pass_offsets``, each blurred by ``aperture / √samples``."""
    offsets = pass_offsets(aperture, samples)
    colors, alphas = 0.0, 0.0
    for offset in offsets:
        c, a = render_thin_lens(
            model,
            camera,
            tau,
            sh_degree,
            cfg,
            focus=focus,
            offset=offset,
            blur_radius=aperture / math.sqrt(samples),
            sh_degree_t=sh_degree_t,
        )
        colors, alphas = colors + c, alphas + a
    return colors / len(offsets), alphas / len(offsets)


def render_thin_lens(
    model: FourDGaussians,
    camera: BatchedCamera,
    tau: float,
    sh_degree: int,
    cfg: RenderingConfig,
    *,
    focus: float,
    offset: tuple[float, float] = (0.0, 0.0),
    blur_radius: float = 0.0,
    sh_degree_t: int = MAX_TEMPORAL_DEGREE,
) -> tuple[Float[Tensor, "1 H W 3"], Float[Tensor, "1 H W 1"]]:
    """One pass: aperture point ``offset`` rendered exactly, every Gaussian blurred by a disk of ``blur_radius``."""
    if cfg.backend != BackendType.ANYCAM4D:
        raise ValueError(f"depth of field renders through the UT kernels; rendering.backend is {cfg.backend.value}")
    if camera.viewmats.shape[0] != 1:
        raise ValueError(f"depth of field renders one camera, got {camera.viewmats.shape[0]}")
    check_render_capability(BackendType.ANYCAM4D, cfg.rasterize_mode)
    check_camera_capability(camera.model, BackendType.ANYCAM4D, camera.distortion)

    args = to_kernel_args(camera)
    splats = model.splats
    appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)
    means2d, depths, radii, colors, means, cov_factors, opacities = ut_4d_depth_of_field_projection_fwd(
        splats["quats_l"].detach(),
        splats["quats_r"].detach(),
        splats["scales_4d_log"].detach(),
        splats["means"].detach(),
        splats["mu_t"].detach(),
        splats["opacities"].detach(),
        appearance.sh0.detach(),
        appearance.shN.detach(),
        appearance.coeffs.detach(),
        sh_degree,
        appearance.kind_id,
        appearance.n_terms,
        appearance.stride,
        args.viewmats,
        args.Ks,
        args.distortion,
        focus,
        blur_radius,
        offset[0],
        offset[1],
        tau,
        model.cov_eps,
        cfg.eps2d,
        cfg.near_plane,
        cfg.far_plane,
        model.prefilter_threshold,
        args.width,
        args.height,
        args.family,
    )

    tile_width = math.ceil(args.width / _TILE_SIZE)
    tile_height = math.ceil(args.height / _TILE_SIZE)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, _TILE_SIZE, tile_width, tile_height, packed=False, n_images=1
    )
    offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height).reshape(1, tile_height, tile_width)
    render_colors, render_alphas, _ = rasterize_from_world_fwd(
        means,
        cov_factors,
        opacities,
        colors,
        args.viewmats,
        args.Ks,
        args.distortion,
        offsets,
        flatten_ids,
        args.width,
        args.height,
        args.family,
        args.valid_mask,
    )
    return render_colors, render_alphas.unsqueeze(-1)
