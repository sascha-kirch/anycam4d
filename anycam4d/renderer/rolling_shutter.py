"""Rolling-shutter forward render through the UT kernels: each image row has its own time and camera pose.

Row ``r`` of an ``H``-row image is exposed at relative readout time ``f_r = (r + 0.5) / H``: at
``tau + (f_r - 0.5) * readout_tau`` and at the pose interpolated between the readout-start and readout-end
viewmats. The frame time is the centre row, so ``readout_tau = 0`` with equal poses is the global-shutter render.
Forward only; blend order is one depth per Gaussian, and colour is evaluated once at ``tau``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from gsplat.cuda._wrapper import isect_offset_encode
from gsplat.cuda._wrapper import isect_tiles
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from anycam4d.cuda.camera import to_kernel_args
from anycam4d.cuda.ext import UT_WORLD_RASTERIZE_TILE_SIZE as _TILE_SIZE
from anycam4d.cuda.ext import rasterize_from_world_rolling_shutter_fwd
from anycam4d.cuda.ext import ut_4d_rolling_shutter_projection_fwd
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


def row_viewmats(
    viewmat_start: Float[Tensor, "4 4"],
    viewmat_end: Float[Tensor, "4 4"],
    height: int,
) -> Float[Tensor, "H 4 4"]:
    """Per-row rigid viewmats: rotation slerped and camera centre lerped at ``(r + 0.5) / H`` (fp64, then cast)."""
    m = torch.stack([viewmat_start, viewmat_end]).detach().double().cpu().numpy()
    rot = m[:, :3, :3]
    centre = -np.einsum("kji,kj->ki", rot, m[:, :3, 3])

    f = (np.arange(height) + 0.5) / height
    rot_rows = Slerp([0.0, 1.0], Rotation.from_matrix(rot))(f).as_matrix()
    centre_rows = (1.0 - f)[:, None] * centre[0] + f[:, None] * centre[1]

    out = np.zeros((height, 4, 4))
    out[:, :3, :3] = rot_rows
    out[:, :3, 3] = -np.einsum("rij,rj->ri", out[:, :3, :3], centre_rows)
    out[:, 3, 3] = 1.0
    return torch.from_numpy(out).to(device=viewmat_start.device, dtype=torch.float32)


@dataclass(frozen=True)
class RollingShutterProjection:
    """Projection outputs: tile box and colour per (camera, Gaussian); conditioned state at ``tau`` per Gaussian."""

    means2d: Float[Tensor, "C N 2"]
    depths: Float[Tensor, "C N"]
    radii: Int[Tensor, "C N 2"]
    colors: Float[Tensor, "C N 3"]
    means: Float[Tensor, "N 3"]
    cov_factors: Float[Tensor, "N 6"]
    base: Float[Tensor, "N"]
    motion: Float[Tensor, "N 4"]
    inv_var_t: Float[Tensor, "N"]
    tau_first: float
    tau_step: float
    camera_args: KernelCameraArgs


def project_rolling_shutter(
    model: FourDGaussians,
    camera: BatchedCamera,
    rows: Float[Tensor, "C H 4 4"],
    tau: float,
    readout_tau: float,
    sh_degree: int,
    cfg: RenderingConfig,
    *,
    sh_degree_t: int = MAX_TEMPORAL_DEGREE,
) -> RollingShutterProjection:
    """Condition at ``tau``; tile box = the tiles each row's global-shutter render assigns on that row."""
    if readout_tau < 0.0:
        raise ValueError(f"readout_tau must be >= 0, got {readout_tau}")
    if cfg.backend != BackendType.ANYCAM4D:
        raise ValueError(f"rolling shutter renders through the UT kernels; rendering.backend is {cfg.backend.value}")
    check_render_capability(BackendType.ANYCAM4D, cfg.rasterize_mode)
    check_camera_capability(camera.model, BackendType.ANYCAM4D, camera.distortion)
    if rows.shape[0] != camera.viewmats.shape[0]:
        raise ValueError(f"rows hold {rows.shape[0]} cameras, camera holds {camera.viewmats.shape[0]}")
    args = to_kernel_args(camera)
    tau_step = readout_tau / args.height
    tau_first = tau + (0.5 / args.height - 0.5) * readout_tau
    splats = model.splats
    appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)
    out = ut_4d_rolling_shutter_projection_fwd(
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
        rows.contiguous(),
        args.Ks,
        args.distortion,
        tau,
        tau_first,
        tau_step,
        model.cov_eps,
        cfg.eps2d,
        cfg.near_plane,
        cfg.far_plane,
        model.prefilter_threshold,
        args.width,
        args.height,
        args.family,
    )
    return RollingShutterProjection(*out, tau_first=tau_first, tau_step=tau_step, camera_args=args)


def render_rolling_shutter(
    model: FourDGaussians,
    camera: BatchedCamera,
    rows: Float[Tensor, "C H 4 4"],
    tau: float,
    readout_tau: float,
    sh_degree: int,
    cfg: RenderingConfig,
    *,
    sh_degree_t: int = MAX_TEMPORAL_DEGREE,
) -> tuple[Float[Tensor, "C H W 3"], Float[Tensor, "C H W 1"]]:
    """Render ``camera`` (intrinsics, lens, size) with per-row viewmats ``rows`` and row times around ``tau``.

    View-dependent colour uses the camera centre midway between the two middle rows: the frame-time centre when
    centres are linear in the row, as ``row_viewmats`` makes them.
    """
    proj = project_rolling_shutter(model, camera, rows, tau, readout_tau, sh_degree, cfg, sh_degree_t=sh_degree_t)
    args = proj.camera_args
    n_cams = rows.shape[0]
    tile_width = math.ceil(args.width / _TILE_SIZE)
    tile_height = math.ceil(args.height / _TILE_SIZE)
    _, isect_ids, flatten_ids = isect_tiles(
        proj.means2d, proj.radii, proj.depths, _TILE_SIZE, tile_width, tile_height, packed=False, n_images=n_cams
    )
    offsets = isect_offset_encode(isect_ids, n_cams, tile_width, tile_height).reshape(n_cams, tile_height, tile_width)

    render_colors, render_alphas = rasterize_from_world_rolling_shutter_fwd(
        proj.means,
        proj.cov_factors,
        proj.base,
        proj.motion,
        proj.inv_var_t,
        proj.colors,
        rows.contiguous(),
        args.Ks,
        args.distortion,
        tau,
        proj.tau_first,
        proj.tau_step,
        cfg.eps2d,
        cfg.near_plane,
        cfg.far_plane,
        model.prefilter_threshold,
        offsets,
        flatten_ids,
        args.width,
        args.height,
        args.family,
        args.valid_mask,
    )
    return render_colors, render_alphas.unsqueeze(-1)
