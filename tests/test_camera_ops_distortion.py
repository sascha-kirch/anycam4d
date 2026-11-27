"""Camera models under strong distortion, in coefficients of real captures.

The reference (``tests.oracles.ut_projection_reference``) pushes the kernel's sigma points through
OpenCV and is checked against Monte-Carlo, so a disagreement localises to the camera model.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from anycam4d.camera.validity import compute_validity
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.enums import CameraModelType
from tests.oracles.factor import cov_of
from tests.oracles.ut_projection_reference import conic_from_cov
from tests.oracles.ut_projection_reference import mc_project
from tests.oracles.ut_projection_reference import project_fisheye
from tests.oracles.ut_projection_reference import ut_project

pytestmark = pytest.mark.skipif(not (HAS_CUDA_EXT and torch.cuda.is_available()), reason="needs the CUDA extension")

PINHOLE_FAMILY = 0
FISHEYE_FAMILY = 1

# Rectilinear rig of the dynopsis calibration room: 24 mm on a 800x600 frame, so the corner sits
# at a normalised radius of 0.94 — far enough out that a strong k1 is doing real work there.
_FISHEYE_GAP = (
    "camera_ops.cuh rejects any fisheye point with cz <= 0, while the inverse direction admits "
    "incidence angles out to THETA_MAX = pi because a real fisheye images past a 180-degree "
    "field. The two therefore disagree about what the lens can see: on the dynopsis 170-degree "
    "lens 13.6% of the frame inverts to an angle beyond 90 degrees, and geometry there is never "
    "assigned a tile even though the rasterizer casts rays that reach it. Lifting the gate (with "
    "near/far culling and the sort key moved from camera-space z to range, since such a primitive "
    "has z <= 0) was tried and made training WORSE — 11.3 dB against 14.4 dB at step 999 on that "
    "scene — so the hemisphere restriction is not the whole story. Reverted; cause still open."
)

RECTILINEAR = dict(width=800, height=600, focal=533.3333, cx=400.0, cy=300.0)
# 170-degree fisheye of the same scene: the corner inverts to an incidence angle of 105 degrees,
# i.e. genuinely behind the camera's z = 0 plane, which a fisheye images and a hemisphere does not.
FISHEYE = dict(width=800, height=800, focal=364.158, cx=400.0, cy=400.0)


def _distortion(k1: float, k2: float, n_slots: int, device: torch.device) -> torch.Tensor:
    """Canonical fixed-width distortion vector carrying a radial pair."""
    dist = torch.zeros(1, n_slots, device=device)
    dist[0, 0] = k1
    dist[0, 1] = k2
    return dist


def _population(n: int, rig: dict, family: int, device: torch.device, seed: int = 0, max_theta: float = 1.2) -> dict:
    """Random population covering the whole frame, corners included, out to field half-angle ``max_theta``.

    Directions are spherical so fields past 90 degrees are reachable; the default keeps sigma points clear of ``cz = 0``.
    """
    generator = torch.Generator().manual_seed(seed)
    depth = torch.empty(n).uniform_(3.0, 7.0, generator=generator)
    if family == FISHEYE_FAMILY:
        theta = torch.empty(n).uniform_(0.0, max_theta, generator=generator)
        phi = torch.empty(n).uniform_(0.0, 2.0 * float(np.pi), generator=generator)
        radial = depth * torch.sin(theta)
        means = torch.stack([radial * torch.cos(phi), radial * torch.sin(phi), depth * torch.cos(theta)], dim=-1)
    else:
        half_w = rig["cx"] / rig["focal"]
        half_h = rig["cy"] / rig["focal"]
        x_n = torch.empty(n).uniform_(-half_w, half_w, generator=generator)
        y_n = torch.empty(n).uniform_(-half_h, half_h, generator=generator)
        means = torch.stack([x_n * depth, y_n * depth, depth], dim=-1)
    quats_l = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator), dim=-1)
    quats_r = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator), dim=-1)
    scales_4d_log = torch.rand(n, 4, generator=generator) * 0.6 - 2.5
    scales_4d_log[:, 3] = float(np.log(0.5))
    return {
        "quats_l": quats_l.to(device),
        "quats_r": quats_r.to(device),
        "scales_4d_log": scales_4d_log.to(device),
        "means": means.to(device),
        "mu_t": torch.zeros(n, device=device),
        "opacities": torch.full((n,), 2.0, device=device),
    }


def _conditioned(pop: dict) -> tuple[np.ndarray, np.ndarray]:
    """Conditioned 3D means and full covariance matrices at tau = 0, on the host."""
    from tests.fused_stages import condition

    means, factors, _, _ = condition(
        pop["quats_l"],
        pop["quats_r"],
        pop["scales_4d_log"],
        pop["means"],
        pop["mu_t"],
        pop["opacities"],
        0.0,
        1e-6,
        0.05,
    )
    return means.cpu().numpy(), cov_of(factors.double()).cpu().numpy()


def _project(pop: dict, rig: dict, dist: torch.Tensor, family: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the UT projection kernel; return (means2d, conics, radii) for the single camera."""
    from tests.fused_stages import fused_ut_project

    device = pop["means"].device
    K = torch.tensor([[[rig["focal"], 0.0, rig["cx"]], [0.0, rig["focal"], rig["cy"]], [0.0, 0.0, 1.0]]], device=device)
    viewmats = torch.eye(4, device=device)[None]
    radii, means2d, _, conics = fused_ut_project(
        pop["quats_l"],
        pop["quats_r"],
        pop["scales_4d_log"],
        pop["means"],
        pop["mu_t"],
        pop["opacities"],
        viewmats,
        K,
        dist,
        0.0,
        1e-6,
        0.3,
        0.01,
        1e10,
        0.05,
        rig["width"],
        rig["height"],
        family,
    )
    return means2d[0].cpu().numpy(), conics[0].cpu().numpy(), radii[0].cpu().numpy()


def _intrinsics(rig: dict) -> np.ndarray:
    return np.array([[rig["focal"], 0.0, rig["cx"]], [0.0, rig["focal"], rig["cy"]], [0.0, 0.0, 1.0]])


def _reference_errors(pop: dict, rig: dict, k1: float, k2: float, family: int) -> tuple[float, float]:
    """Worst-case (mean2d pixels, relative conic) disagreement against the host UT reference."""
    device = pop["means"].device
    n_slots = 12 if family == PINHOLE_FAMILY else 4
    means2d, conics, radii = _project(pop, rig, _distortion(k1, k2, n_slots, device), family)
    means_3d_cond, covars3d = _conditioned(pop)
    K = _intrinsics(rig)
    dist = np.zeros(12)
    dist[0], dist[1] = k1, k2
    worst_mean, worst_conic = 0.0, 0.0
    for i in range(len(means2d)):
        if radii[i].min() <= 0:  # culled by the kernel; nothing to compare
            continue
        ref_mean, ref_cov = ut_project(means_3d_cond[i], covars3d[i], K, dist, fisheye=family == FISHEYE_FAMILY)
        ref_cov = ref_cov + np.eye(2) * 0.3  # the kernel's eps2d blur
        ref_conic = conic_from_cov(ref_cov)
        worst_mean = max(worst_mean, float(np.hypot(*(means2d[i] - ref_mean))))
        worst_conic = max(worst_conic, float(np.abs(ref_conic - conics[i]).max() / np.abs(ref_conic).max()))
    return worst_mean, worst_conic


@pytest.mark.parametrize(
    ("k1", "k2"),
    [(0.0115, -0.0058), (0.0863, -0.1408), (0.1, 0.0)],
    ids=["dynerf", "selfcap-bike", "mild-barrel"],
)
def test_ut_projection_matches_reference_under_mild_distortion(k1: float, k2: float) -> None:
    """The pinhole kernel matches the reference at float precision on mild, real coefficients."""
    pop = _population(256, RECTILINEAR, PINHOLE_FAMILY, torch.device("cuda"))
    worst_mean, worst_conic = _reference_errors(pop, RECTILINEAR, k1, k2, PINHOLE_FAMILY)
    assert worst_mean < 1e-3, f"mean2d disagreement {worst_mean:.3e} px"
    assert worst_conic < 1e-4, f"conic disagreement {worst_conic:.3e}"


@pytest.mark.parametrize(
    ("k1", "k2"), [(0.0, 0.0), (0.1001, -0.0183), (-0.25, 0.06)], ids=["equidistant", "truck", "dynopsis"]
)
def test_ut_projection_matches_reference_on_fisheye(k1: float, k2: float) -> None:
    """The fisheye kernel matches the reference inside the hemisphere, from equidistant to strong barrel."""
    pop = _population(384, FISHEYE, FISHEYE_FAMILY, torch.device("cuda"))
    worst_mean, worst_conic = _reference_errors(pop, FISHEYE, k1, k2, FISHEYE_FAMILY)
    assert worst_mean < 1e-3, f"mean2d disagreement {worst_mean:.3e} px"
    assert worst_conic < 1e-3, f"conic disagreement {worst_conic:.3e}"


@pytest.mark.xfail(strict=True, reason=_FISHEYE_GAP)
def test_ut_projection_matches_reference_across_the_full_fisheye_field() -> None:
    """A 170-degree lens transforms correctly out to 105 degrees; a dropped sigma point would wreck the fit."""
    pop = _population(384, FISHEYE, FISHEYE_FAMILY, torch.device("cuda"), max_theta=1.84)
    worst_mean, worst_conic = _reference_errors(pop, FISHEYE, -0.25, 0.06, FISHEYE_FAMILY)
    assert worst_mean < 1e-3, f"mean2d disagreement {worst_mean:.3e} px"
    assert worst_conic < 1e-3, f"conic disagreement {worst_conic:.3e}"


def test_fisheye_reference_agrees_with_opencv_where_opencv_is_valid() -> None:
    """The Kannala-Brandt reference matches ``cv2.fisheye`` inside 60 degrees, where OpenCV's ``atan(|xy|/z)`` is sound."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, np.radians(60.0), size=512)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=512)
    distance = rng.uniform(3.0, 7.0, size=512)
    points = np.stack(
        [distance * np.sin(theta) * np.cos(phi), distance * np.sin(theta) * np.sin(phi), distance * np.cos(theta)],
        axis=-1,
    )
    K = _intrinsics(FISHEYE)
    dist = np.zeros(12)
    dist[0], dist[1] = -0.25, 0.06
    ours = project_fisheye(points, K, dist)
    theirs = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist[:4].reshape(4, 1))[
        0
    ].reshape(-1, 2)
    assert np.abs(ours - theirs).max() < 1e-6, f"reference drifts from OpenCV by {np.abs(ours - theirs).max():.3e} px"


def test_unscented_transform_tracks_the_true_projected_distribution() -> None:
    """The unscented transform stays close to Monte-Carlo at ``k1 = -0.28``, so kernel failures are kernel results."""
    pop = _population(64, RECTILINEAR, PINHOLE_FAMILY, torch.device("cuda"))
    means_3d_cond, covars3d = _conditioned(pop)
    K = _intrinsics(RECTILINEAR)
    dist = np.zeros(12)
    dist[0], dist[1] = -0.28, 0.06
    worst_mean, worst_cov = 0.0, 0.0
    for i in range(len(means_3d_cond)):
        ref_mean, ref_cov = ut_project(means_3d_cond[i], covars3d[i], K, dist, fisheye=False)
        mc_mean, mc_cov = mc_project(means_3d_cond[i], covars3d[i], K, dist, fisheye=False)
        worst_mean = max(worst_mean, float(np.hypot(*(ref_mean - mc_mean))))
        worst_cov = max(worst_cov, float(np.abs(ref_cov - mc_cov).max() / np.abs(mc_cov).max()))
    assert worst_mean < 1.0, f"UT mean is {worst_mean:.2f} px off the sampled mean"
    assert worst_cov < 0.25, f"UT covariance is {worst_cov:.1%} off the sampled covariance"


def test_pinhole_projection_admits_every_physically_valid_point() -> None:
    """Points a real lens images are not dropped; losing corner sigma points shows up as a tens-of-pixels mean shift."""
    pop = _population(256, RECTILINEAR, PINHOLE_FAMILY, torch.device("cuda"))
    worst_mean, _ = _reference_errors(pop, RECTILINEAR, -0.28, 0.06, PINHOLE_FAMILY)
    assert worst_mean < 1e-3, f"mean2d disagreement {worst_mean:.3f} px"


@pytest.mark.xfail(strict=True, reason=_FISHEYE_GAP)
def test_fisheye_projection_admits_points_beyond_the_hemisphere() -> None:
    """Projection accepts every ray unprojection generates."""
    from tests.fused_stages import fused_ut_project  # noqa: F401  (import guard only)

    device = torch.device("cuda")
    # 100 degrees off axis on the image diagonal: this lens only images past 90 degrees towards the corners.
    theta = float(np.radians(100.0))
    phi = float(np.radians(45.0))
    distance = 5.0
    mean = torch.tensor(
        [
            [
                distance * float(np.sin(theta) * np.cos(phi)),
                distance * float(np.sin(theta) * np.sin(phi)),
                distance * float(np.cos(theta)),
            ]
        ],
        device=device,
    )
    pop = {
        "quats_l": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "scales_4d_log": torch.tensor([[-2.5, -2.5, -2.5, float(np.log(0.5))]], device=device),
        "means": mean,
        "mu_t": torch.zeros(1, device=device),
        "opacities": torch.full((1,), 2.0, device=device),
    }
    _, _, radii = _project(pop, FISHEYE, _distortion(-0.25, 0.06, 4, device), FISHEYE_FAMILY)
    assert radii[0].min() > 0, "a Gaussian inside a 170-degree field was culled by the hemisphere gate"


@pytest.mark.parametrize(("k1", "k2"), [(0.1, 0.0), (-0.25, 0.06), (-0.28, 0.06), (-0.35, 0.09)])
def test_host_validity_is_all_true_where_the_lens_is_monotone(k1: float, k2: float) -> None:
    """Strong but monotone barrel distortion observes every pixel."""
    K = torch.tensor([[533.3333, 0.0, 400.0], [0.0, 533.3333, 300.0], [0.0, 0.0, 1.0]])
    dist = torch.zeros(12)
    dist[0], dist[1] = k1, k2
    valid = compute_validity(CameraModelType.OPENCV, K, dist, 800, 600)
    assert valid.all(), f"host footprint dropped {(~valid).float().mean():.2%} of the frame"


def _render_loss(model, camera, tau: float = 0.0) -> torch.Tensor:
    """Scalar loss over a UT render — the quantity training actually differentiates."""
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.enums import BackendType
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend

    backend = UTFusedBackend(RenderingConfig(backend=BackendType.ANYCAM4D))
    out = backend.render(model, camera, tau=tau, sh_degree=1)
    return out.colors.square().sum()


def _distorted_camera(model_type: CameraModelType, k1: float, k2: float, device: torch.device):
    from anycam4d.camera.batched import BatchedCamera

    width = height = 64
    viewmats = torch.eye(4, device=device)[None]
    viewmats[:, 2, 3] = 5.0
    Ks = torch.tensor([[[60.0, 0.0, width / 2.0], [0.0, 60.0, height / 2.0], [0.0, 0.0, 1.0]]], device=device)
    n_slots = 4 if model_type is CameraModelType.FISHEYE_OPENCV else 12
    return BatchedCamera(viewmats, Ks, model_type, width, height, _distortion(k1, k2, n_slots, device))


@pytest.mark.parametrize(
    ("model_type", "k1", "k2"),
    [
        (CameraModelType.OPENCV, -0.28, 0.06),
        (CameraModelType.OPENCV, -0.35, 0.09),
        (CameraModelType.FISHEYE_OPENCV, -0.25, 0.06),
    ],
)
def test_render_gradients_are_finite_under_strong_distortion(
    random_model, model_type: CameraModelType, k1: float, k2: float
) -> None:
    """Backward through a strongly distorted lens stays finite."""
    device = torch.device("cuda")
    model = random_model.to(device)
    camera = _distorted_camera(model_type, k1, k2, device)
    _render_loss(model, camera).backward()
    for name, param in model.splats.items():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def _finite_difference_disagreement(model, camera, n_probe: int = 24, step: float = 1e-3) -> float:
    """Median relative gap between analytic mean gradient and central difference over the ``n_probe`` largest gradients."""
    means = model.splats["means"]
    model.zero_grad(set_to_none=True)
    _render_loss(model, camera).backward()
    analytic = means.grad.clone()
    probes = analytic.abs().sum(dim=1).argsort(descending=True)[:n_probe]

    gaps = []
    with torch.no_grad():
        for index in probes.tolist():
            numeric = torch.zeros(3, device=means.device)
            for axis in range(3):
                original = means[index, axis].item()
                means[index, axis] = original + step
                plus = _render_loss(model, camera).item()
                means[index, axis] = original - step
                minus = _render_loss(model, camera).item()
                means[index, axis] = original
                numeric[axis] = (plus - minus) / (2.0 * step)
            scale = max(float(analytic[index].abs().max()), float(numeric.abs().max()), 1e-9)
            gaps.append(float((analytic[index] - numeric).abs().max()) / scale)
    return float(np.median(gaps))


@pytest.mark.parametrize(
    ("model_type", "k1", "k2"),
    [
        (CameraModelType.OPENCV, 0.1, 0.0),
        (CameraModelType.OPENCV, -0.28, 0.06),
        (CameraModelType.FISHEYE_OPENCV, -0.25, 0.06),
    ],
)
def test_distortion_does_not_degrade_the_gradient(
    random_model, model_type: CameraModelType, k1: float, k2: float
) -> None:
    """Distortion does not make the analytic gradient less faithful than the undistorted baseline.

    Finite differences through the rasterizer already disagree by ~8% on a plain pinhole, so the
    baseline measured in the same run is the yardstick, not an absolute threshold.
    """
    device = torch.device("cuda")
    model = random_model.to(device)
    baseline = _finite_difference_disagreement(model, _distorted_camera(CameraModelType.PINHOLE, 0.0, 0.0, device))
    distorted = _finite_difference_disagreement(model, _distorted_camera(model_type, k1, k2, device))
    assert distorted < max(3.0 * baseline, 0.25), (
        f"distorted gap {distorted:.3f} vs undistorted baseline {baseline:.3f}"
    )
