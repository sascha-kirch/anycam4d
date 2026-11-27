"""The world rasterizer's per-pixel ray-validity gate.

The kernel composites nothing along a ray outside the lens's monotone domain. It must stay a subset of
the host mask the loss reduces over. Validity is probed with one huge Gaussian that every valid ray hits.
"""

from __future__ import annotations

import math

import pytest
import torch

from anycam4d.camera.valid_mask import pack_validity
from anycam4d.camera.validity import compute_validity
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import ProjectionFamily
from anycam4d.enums import CameraModelType


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="requires the anycam4d CUDA extension")

TILE = 16


def _probe_scene(device: torch.device, W: int, H: int, *, n_dist: int):
    """One huge Gaussian covering every tile — hit by any ray the camera model inverts."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles

    tile_h, tile_w = math.ceil(H / TILE), math.ceil(W / TILE)
    means_3d_cond = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    # Isotropic Σ = 100²·I: D² ≈ 0 for every ray direction, so alpha is opacity-limited
    # rather than geometry-limited and the only thing that can zero it is the gate.
    factors = torch.tensor([[1e2, 0.0, 0.0, 1e2, 0.0, 1e2]], device=device)  # Σ = 1e4·I
    opacities = torch.tensor([0.9], device=device)
    colors = torch.ones(1, 1, 3, device=device)

    means2d = torch.zeros(1, 1, 2, device=device)
    depths = torch.ones(1, 1, device=device)
    radii = torch.full((1, 1, 2), max(W, H) * 4, dtype=torch.int32, device=device)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, TILE, tile_w, tile_h, packed=False, n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h).reshape(1, tile_h, tile_w)
    dist = torch.zeros((1, n_dist), device=device)
    return means_3d_cond, factors, opacities, colors, offsets.contiguous(), flatten_ids.contiguous(), dist


def _kernel_valid_mask(
    K: torch.Tensor,
    family: ProjectionFamily,
    W: int,
    H: int,
    *,
    n_dist: int,
    distortion: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[H, W]`` bool mask of pixels the kernel composited (i.e. whose ray inverted)."""
    from anycam4d.cuda.ext import rasterize_from_world_fwd

    device = torch.device("cuda")
    means_3d_cond, factors, opacities, colors, offsets, flatten_ids, dist = _probe_scene(device, W, H, n_dist=n_dist)
    if distortion is not None:
        dist = distortion.to(device).reshape(1, n_dist).contiguous()
    viewmat = torch.eye(4, device=device).unsqueeze(0).contiguous()
    _rc, ra, last = rasterize_from_world_fwd(
        means_3d_cond,
        factors,
        opacities,
        colors,
        viewmat,
        K.to(device).unsqueeze(0).contiguous(),
        dist,
        offsets,
        flatten_ids,
        W,
        H,
        int(family),
        None if valid_mask is None else pack_validity(valid_mask.to(device)).unsqueeze(0).contiguous(),
    )
    assert torch.isfinite(ra).all(), "gated pixels must be background, never NaN/Inf"
    return (ra[0] > 0.0) & (last[0] >= 0)


def _past_half_turn_fisheye_K(W: int, H: int) -> torch.Tensor:
    """Equidistant f = 12 on a 64-pixel frame: the corners ask for θ ≈ 3.77 rad > π."""
    return torch.tensor([[12.0, 0.0, W / 2.0], [0.0, 12.0, H / 2.0], [0.0, 0.0, 1.0]])


def test_fisheye_gate_fires_past_a_half_turn():
    """Equidistant at f = 12: frame corners ask for θ ≈ 3.77 > π and are gated."""
    W = H = 64
    K = torch.tensor([[12.0, 0.0, W / 2.0], [0.0, 12.0, H / 2.0], [0.0, 0.0, 1.0]])
    valid = _kernel_valid_mask(K, ProjectionFamily.FISHEYE_FAMILY, W, H, n_dist=4)

    assert not valid.all(), "fixture must actually leave the physical angle range"
    assert valid[H // 2, W // 2], "the optical axis always inverts"
    assert not valid[0, 0], "the corner is past a half turn"


def test_fisheye_inside_the_physical_domain_is_never_gated():
    """Lenses that fold or reach π outside their frame composite every pixel."""
    W = H = 32
    K = torch.tensor([[30.0, 0.0, W / 2.0], [0.0, 30.0, H / 2.0], [0.0, 0.0, 1.0]])
    for dist in (torch.zeros(4), torch.tensor([-0.05, 0.0, 0.0, 0.0])):
        valid = _kernel_valid_mask(K, ProjectionFamily.FISHEYE_FAMILY, W, H, n_dist=4, distortion=dist)
        assert valid.all(), f"gate fired for distortion {dist.tolist()}"


def test_fisheye_gate_fires_past_the_fold():
    """k1 = -0.15 folds at ~13 px inside the frame, so part of the frame is gated.

    The kernel tests at an unconverged Newton iterate, so the gated set is a fraction, not an annulus.
    """
    W = H = 64
    K = torch.tensor([[12.0, 0.0, W / 2.0], [0.0, 12.0, H / 2.0], [0.0, 0.0, 1.0]])
    dist = torch.tensor([-0.15, 0.0, 0.0, 0.0])
    valid = _kernel_valid_mask(K, ProjectionFamily.FISHEYE_FAMILY, W, H, n_dist=4, distortion=dist)

    assert valid[H // 2, W // 2], "the optical axis always inverts"
    assert float(valid.float().mean()) < 0.5, "most of a frame this far past the fold must be gated"


def test_pinhole_family_gates_its_fold():
    """A bare pinhole is ungated; k1 = -0.3 folds inside the frame and part of it is gated."""
    W = H = 64
    K = torch.tensor([[35.0, 0.0, W / 2.0], [0.0, 35.0, H / 2.0], [0.0, 0.0, 1.0]])
    valid = _kernel_valid_mask(K, ProjectionFamily.PINHOLE_FAMILY, W, H, n_dist=12, distortion=torch.zeros(12))
    assert valid.all(), "an undistorted pinhole must never gate"

    folded = torch.zeros(12)
    folded[0] = -0.3
    valid = _kernel_valid_mask(K, ProjectionFamily.PINHOLE_FAMILY, W, H, n_dist=12, distortion=folded)
    assert valid[H // 2, W // 2], "the optical axis always inverts"
    assert float(valid.float().mean()) < 0.7, "a lens folding this far inside its frame must gate"


def test_gated_pixels_contribute_no_gradient():
    """Gradient placed only on gated pixels reaches no parameter."""
    from anycam4d.cuda.autograd import RasterizeWorldFunction

    device = torch.device("cuda")
    W = H = 64
    K = _past_half_turn_fisheye_K(W, H).to(device)
    means_3d_cond, factors, opacities, colors, offsets, flatten_ids, dist = _probe_scene(device, W, H, n_dist=4)
    for t in (means_3d_cond, factors, opacities, colors):
        t.requires_grad_(True)
    viewmat = torch.eye(4, device=device).unsqueeze(0).contiguous()

    out = RasterizeWorldFunction.apply(
        means_3d_cond,
        factors,
        opacities,
        colors,
        viewmat,
        K.unsqueeze(0).contiguous(),
        dist,
        offsets,
        flatten_ids,
        W,
        H,
        int(ProjectionFamily.FISHEYE_FAMILY),
    )
    rc, ra = out[0], out[1]
    gated = ~(ra[0].detach() > 0.0)
    assert gated.any(), "fixture must actually contain gated pixels"

    # Weight the loss by the gated mask alone — a correct gate leaves nothing to differentiate.
    (rc[0] * gated[..., None]).sum().backward()
    for name, t in (("means", means_3d_cond), ("factors", factors), ("opacities", opacities), ("colors", colors)):
        assert torch.count_nonzero(t.grad) == 0, f"gated pixels leaked gradient into {name}"


def test_valid_pixels_still_receive_gradient():
    """The same scene, weighted by the *complement*, produces the usual nonzero gradient."""
    from anycam4d.cuda.autograd import RasterizeWorldFunction

    device = torch.device("cuda")
    W = H = 64
    K = _past_half_turn_fisheye_K(W, H).to(device)
    means_3d_cond, factors, opacities, colors, offsets, flatten_ids, dist = _probe_scene(device, W, H, n_dist=4)
    for t in (means_3d_cond, factors, opacities, colors):
        t.requires_grad_(True)
    viewmat = torch.eye(4, device=device).unsqueeze(0).contiguous()

    out = RasterizeWorldFunction.apply(
        means_3d_cond,
        factors,
        opacities,
        colors,
        viewmat,
        K.unsqueeze(0).contiguous(),
        dist,
        offsets,
        flatten_ids,
        W,
        H,
        int(ProjectionFamily.FISHEYE_FAMILY),
    )
    out[0].sum().backward()
    assert torch.count_nonzero(colors.grad) > 0
    assert torch.isfinite(colors.grad).all()
    assert torch.isfinite(means_3d_cond.grad).all()


def _assert_safe_subset(kernel_valid: torch.Tensor, host_valid: torch.Tensor, label: str) -> int:
    """kernel-gated ⊆ host-invalid. Returns the disagreement (host-valid but kernel-gated) count."""
    violations = int(((~kernel_valid.cpu()) & host_valid).sum())
    assert violations == 0, f"{label}: kernel gates {violations} pixels the host counts as observed"
    return int(((~host_valid) & kernel_valid.cpu()).sum())


@pytest.mark.parametrize("res", [(64, 64), (96, 72)])
@pytest.mark.parametrize("k1", [-0.15, -0.05, 0.0])
def test_fisheye_gate_is_subset_of_host_validity(res, k1):
    """Fisheye kernel gate never exceeds the host footprint (fp32 Newton iterate vs fp64 solve)."""
    W, H = res
    fx = fy = 12.0 * W / 64.0
    K = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
    dist4 = torch.tensor([k1, 0.0, 0.0, 0.0])
    kernel_valid = _kernel_valid_mask(K, ProjectionFamily.FISHEYE_FAMILY, W, H, n_dist=4, distortion=dist4)
    host_valid = compute_validity(CameraModelType.FISHEYE_OPENCV, K, dist4, W, H)
    _assert_safe_subset(kernel_valid, host_valid, f"FISHEYE k1={k1} {W}x{H}")


@pytest.mark.parametrize("res", [(64, 64), (96, 72)])
@pytest.mark.parametrize("k1", [-0.5, -0.3, 0.2])
def test_pinhole_gate_is_subset_of_host_validity(res, k1):
    """Distorted pinhole kernel gate never exceeds the host footprint."""
    W, H = res
    fx = fy = 35.0 * W / 64.0
    K = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
    dist12 = torch.zeros(12)
    dist12[0] = k1  # the Brown-Conrady fold at −0.3
    kernel_valid = _kernel_valid_mask(K, ProjectionFamily.PINHOLE_FAMILY, W, H, n_dist=12, distortion=dist12)
    host_valid = compute_validity(CameraModelType.OPENCV, K, dist12, W, H)
    _assert_safe_subset(kernel_valid, host_valid, f"OPENCV k1={k1} {W}x{H}")


@pytest.mark.parametrize("res", [(64, 64), (96, 72)])
def test_unfolded_pinhole_gate_equals_host_validity(res):
    """A lens without a turning point is fully observed on both sides, exactly."""
    W, H = res
    fx = fy = 35.0 * W / 64.0
    K = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
    for k1 in (0.0, 0.2, 0.5):
        dist12 = torch.zeros(12)
        dist12[0] = k1
        kernel_valid = _kernel_valid_mask(K, ProjectionFamily.PINHOLE_FAMILY, W, H, n_dist=12, distortion=dist12)
        host_valid = compute_validity(CameraModelType.OPENCV, K, dist12, W, H)
        assert torch.equal(kernel_valid.cpu(), host_valid), f"k1={k1} {W}x{H}"


@pytest.mark.parametrize("res", [(64, 64), (96, 72)])
def test_equidistant_fisheye_gate_equals_host_validity(res):
    """An equidistant lens is bounded at θ = π on both sides; the focal length puts that circle in frame."""
    W, H = res
    fx = fy = 12.0 * W / 64.0  # corner at θ ≈ 3.8 rad, so the bound is inside the frame
    K = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
    dist4 = torch.zeros(4)
    kernel_valid = _kernel_valid_mask(K, ProjectionFamily.FISHEYE_FAMILY, W, H, n_dist=4, distortion=dist4)
    host_valid = compute_validity(CameraModelType.FISHEYE_EQUIDISTANT, K, dist4, W, H)

    assert not kernel_valid.all(), "fixture must actually cross the bound"
    assert torch.equal(kernel_valid.cpu(), host_valid)


@pytest.mark.parametrize("k1", [-0.2, -0.3, -0.5])
@pytest.mark.parametrize("res", [(64, 64), (96, 72)])
def test_pinhole_uploaded_host_mask_wins_over_the_looser_gate(res, k1):
    """With the host mask uploaded, the rendered footprint is exactly the host mask."""
    W, H = res
    K = torch.tensor([[35.0 * W / 64.0, 0.0, W / 2.0], [0.0, 35.0 * W / 64.0, H / 2.0], [0.0, 0.0, 1.0]])
    dist12 = torch.zeros(12)
    dist12[0] = k1
    host_valid = compute_validity(CameraModelType.OPENCV, K, dist12, W, H)
    gate_only = _kernel_valid_mask(K, ProjectionFamily.PINHOLE_FAMILY, W, H, n_dist=12, distortion=dist12)

    with_mask = _kernel_valid_mask(
        K, ProjectionFamily.PINHOLE_FAMILY, W, H, n_dist=12, distortion=dist12, valid_mask=host_valid
    )

    assert not host_valid.all(), "fixture must have unobserved pixels"
    assert torch.equal(with_mask.cpu(), host_valid)
    assert (gate_only.cpu() & ~host_valid).any(), "the gate alone must be looser here, or this proves nothing"
