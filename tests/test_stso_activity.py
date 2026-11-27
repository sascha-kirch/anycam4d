"""``build_activity_mask`` reads the active set from render meta, and it equals the gradient-receiving set."""

from __future__ import annotations

import pytest
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.enums import BackendType
from anycam4d.enums import CameraModelType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.trainer.activity import build_activity_mask


_needs_cuda = pytest.mark.skipif(not HAS_CUDA_EXT, reason="render-based identity runs on the CUDA kernels")

_H = _W = 32
_EPS_T = 0.05


def test_missing_meta_all_true() -> None:
    """No gaussian_ids/radii → all-ones (eval / NullStrategy path)."""
    dev = torch.device("cpu")
    assert build_activity_mask({}, 5, dev).equal(torch.ones(5, dtype=torch.bool))
    assert build_activity_mask(None, 5, dev).equal(torch.ones(5, dtype=torch.bool))
    # packed flag set but gaussian_ids None → not a packed source → falls through to all-True.
    assert build_activity_mask({"packed": True, "gaussian_ids": None}, 5, dev).all()


def test_dense_radii_shape_reduction() -> None:
    """A hand-built radii [C, N, 2] reduces to [N] via OR over cameras (guards the .any(dim=0) bug)."""
    dev = torch.device("cpu")
    n = 4
    radii = torch.zeros(2, n, 2)  # [C=2, N=4, 2]
    radii[0, 1, 0] = 3  # cam0 sees gaussian 1
    radii[1, 3, 1] = 5  # cam1 sees gaussian 3
    mask = build_activity_mask({"radii": radii, "packed": False, "gaussian_ids": None}, n, dev)
    assert mask.shape == (n,)
    assert mask.dtype == torch.bool
    assert mask.equal(torch.tensor([False, True, False, True]))


def test_dense_radii_2d_layout() -> None:
    """A [C, N] radii (no trailing width/height axis) still reduces to [N]."""
    dev = torch.device("cpu")
    radii = torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])  # [C=2, N=3]
    mask = build_activity_mask({"radii": radii}, 3, dev)
    assert mask.equal(torch.tensor([True, True, False]))


def test_packed_scatter_union() -> None:
    """Hand-built gaussian_ids across multiple cameras → correct [N] union."""
    dev = torch.device("cpu")
    n = 6
    # Two cameras' packed rows: cam0 lit {0,2}, cam1 lit {2,5}. Union = {0,2,5}.
    gaussian_ids = torch.tensor([0, 2, 2, 5], dtype=torch.long)
    mask = build_activity_mask({"packed": True, "gaussian_ids": gaussian_ids}, n, dev)
    assert mask.equal(torch.tensor([True, False, True, False, False, True]))


def _pinhole_camera(device: torch.device, viewmats: torch.Tensor) -> BatchedCamera:
    """A pinhole BatchedCamera for the given world-to-cam matrices [C, 4, 4]."""
    c = viewmats.shape[0]
    fx = fy = 50.0
    cx, cy = (_W - 1) / 2.0, (_H - 1) / 2.0
    k = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device)
    ks = k.unsqueeze(0).expand(c, -1, -1).contiguous()
    return BatchedCamera(viewmats, ks, CameraModelType.PINHOLE, _W, _H, torch.zeros(c, 12, device=device))


def _front_view(device: torch.device) -> torch.Tensor:
    """Single world-to-cam matrix: camera at origin looking down +z."""
    return torch.eye(4, device=device).unsqueeze(0)


def _mixed_model(device: torch.device, tau: float) -> tuple[FourDGaussians, int]:
    """Three groups of ``g``: active, temporally gated, and behind the camera; returns ``(model, g)``."""
    g = 8
    n = 3 * g
    model = FourDGaussians.from_random(n, sh_degree=1).to(device)
    model.prefilter_threshold = _EPS_T
    depths = torch.linspace(2.0, 5.0, g, device=device)
    angles = torch.linspace(-0.15, 0.15, g, device=device)
    front = torch.stack([torch.tan(angles) * depths, torch.zeros_like(depths), depths], dim=-1)
    behind = front.clone()
    behind[:, 2] = -depths  # move group C behind the camera
    means = torch.cat([front, front.clone(), behind], dim=0)
    mu_t = torch.full((n,), tau, device=device)
    mu_t[g : 2 * g] = tau + 5.0  # group B temporally gated (large |τ − μ_t|)
    with torch.no_grad():
        model.splats["means"].copy_(means)
        model.splats["mu_t"].copy_(mu_t)
        model.splats["opacities"].fill_(3.0)
        model.splats["sh0"].normal_(0.0, 0.3)
    return model, g


def _grad_active_set(model: FourDGaussians, n: int, device: torch.device) -> torch.Tensor:
    """Per-Gaussian gradient-receiving set: True where any splat param received nonzero grad."""
    active = torch.zeros(n, dtype=torch.bool, device=device)
    for p in model.splats.values():
        if p.grad is not None:
            g = p.grad.reshape(n, -1)
            active |= (g != 0).any(dim=-1)
    return active


def _make_backend(backend_type: BackendType, packed: bool):  # noqa: ANN202 — test helper
    return RenderingConfig(backend=backend_type, packed=packed)


_IDENTITY_CASES = [
    (BackendType.EWA, "ewa"),
    (BackendType.ANYCAM4D, "ut"),
]


@_needs_cuda
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(("backend_type", "name"), _IDENTITY_CASES)
def test_activity_mask_equals_gradient_set(backend_type: BackendType, name: str, packed: bool) -> None:
    """build_activity_mask(meta) == gradient set == analytic (vˢ ∧ [w_t>ε_t]) for every backend/path."""
    from anycam4d.renderer.factory import BackendFactory

    device = torch.device("cuda")
    tau = 0.3
    model, g = _mixed_model(device, tau)
    n = 3 * g
    views = _front_view(device)
    camera = _pinhole_camera(device, views)
    backend = BackendFactory.create(_make_backend(backend_type, packed))

    out = backend.render(model, camera, tau, model.sh_degree)
    mask = build_activity_mask(out.meta, n, device)

    (out.colors.sum() + out.alphas.sum()).backward()
    grad_set = _grad_active_set(model, n, device)

    # Core identity: the mask is exactly the gradient-receiving set.
    assert mask.equal(grad_set), f"{name} packed={packed}: mask != gradient set"

    # Analytic cross-check: group A active, groups B (temporal) and C (spatial) inactive.
    assert mask[:g].all(), f"{name}: front temporally-centred group should be active"
    assert not mask[g : 2 * g].any(), f"{name}: temporally-gated group should be inactive"
    assert not mask[2 * g :].any(), f"{name}: spatially-culled group should be inactive"


@_needs_cuda
def test_c_gt_1_mask_is_or_over_cameras() -> None:
    """A frame-aligned C>1 batch ORs over cameras: each camera sees a disjoint group, union = all."""
    from anycam4d.renderer.factory import BackendFactory

    device = torch.device("cuda")
    tau = 0.3
    # Two groups, no temporal gating: group F in front of cam0, group B behind cam0 (front of cam1).
    g = 8
    n = 2 * g
    model = FourDGaussians.from_random(n, sh_degree=1).to(device)
    model.prefilter_threshold = _EPS_T
    depths = torch.linspace(2.0, 5.0, g, device=device)
    angles = torch.linspace(-0.15, 0.15, g, device=device)
    front = torch.stack([torch.tan(angles) * depths, torch.zeros_like(depths), depths], dim=-1)
    back = front.clone()
    back[:, 2] = -depths
    with torch.no_grad():
        model.splats["means"].copy_(torch.cat([front, back], dim=0))
        model.splats["mu_t"].fill_(tau)
        model.splats["opacities"].fill_(3.0)

    # cam0: identity (looks +z, sees F). cam1: 180° about y (looks −z, sees B).
    v0 = torch.eye(4, device=device)
    v1 = torch.eye(4, device=device)
    v1[0, 0] = -1.0
    v1[2, 2] = -1.0
    camera = _pinhole_camera(device, torch.stack([v0, v1], dim=0))
    backend = BackendFactory.create(_make_backend(BackendType.EWA, packed=False))

    with torch.no_grad():
        out = backend.render(model, camera, tau, model.sh_degree)

    radii = out.meta["radii"]  # [C=2, N, 2]
    cam0_seen = (radii[0] > 0).any(dim=-1)
    cam1_seen = (radii[1] > 0).any(dim=-1)
    # Genuinely disjoint: AND would be empty, OR is the union.
    assert not (cam0_seen & cam1_seen).any(), "groups must be disjoint per camera to test OR vs AND"
    mask = build_activity_mask(out.meta, n, device)
    assert mask.equal(cam0_seen | cam1_seen)
    assert mask.all(), "union of both cameras should cover every Gaussian"
