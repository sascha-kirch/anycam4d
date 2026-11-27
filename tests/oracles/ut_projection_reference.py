"""Host reference for the UT camera projection, and its Monte-Carlo ground truth.

``ut_project`` mirrors ``ut_constants.cuh`` but projects through OpenCV, so a kernel disagreement
isolates the camera model. ``mc_project`` checks the transform itself: failing together means the
unscented transform is inadequate at that distortion, not that the kernel is wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from jaxtyping import Float

# ut_constants.cuh, (alpha, beta, kappa) = (1, 2, 0), state dimension 3:
#   lambda = alpha^2 (3 + kappa) - 3 = 0,  c_spread = sqrt(3 + lambda) = sqrt(3)
#   W_m = [0, 1/6 x6],  W_c = [2, 1/6 x6]
C_SPREAD: float = float(np.sqrt(3.0))
WEIGHTS_MEAN: Float[np.ndarray, "7"] = np.array([0.0] + [0.5 / 3.0] * 6)
WEIGHTS_COV: Float[np.ndarray, "7"] = np.array([2.0] + [0.5 / 3.0] * 6)


def sigma_points(mean: Float[np.ndarray, "3"], cov: Float[np.ndarray, "3 3"]) -> Float[np.ndarray, "7 3"]:
    """The seven scaled-UT sigma points ``[7, 3]``, ordered like the kernel: centre, +e0..e2, -e0..e2."""
    chol = np.linalg.cholesky(cov)  # lower triangular; its columns are the offset directions
    offsets = C_SPREAD * chol.T
    return np.concatenate([mean[None, :], mean + offsets, mean - offsets], axis=0)


def project_fisheye(
    points_cam: Float[np.ndarray, "N 3"],
    K: Float[np.ndarray, "3 3"],
    distortion: Float[np.ndarray, "D"],
) -> Float[np.ndarray, "N 2"]:
    """Kannala-Brandt projection to ``[N, 2]`` pixels over the whole field.

    Uses ``atan2`` where ``cv2.fisheye`` uses ``atan(|xy|/z)``, which folds points behind the camera plane.
    """
    k1, k2, k3, k4 = (float(distortion[i]) for i in range(4))
    radial = np.hypot(points_cam[:, 0], points_cam[:, 1])
    theta = np.arctan2(radial, points_cam[:, 2])
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(radial > 1e-12, theta_d / np.where(radial > 1e-12, radial, 1.0), 1.0)
    u = K[0, 0] * scale * points_cam[:, 0] + K[0, 2]
    v = K[1, 1] * scale * points_cam[:, 1] + K[1, 2]
    return np.stack([u, v], axis=-1)


def project_opencv(
    points_cam: Float[np.ndarray, "N 3"],
    K: Float[np.ndarray, "3 3"],
    distortion: Float[np.ndarray, "D"],
    *,
    fisheye: bool,
) -> Float[np.ndarray, "N 2"]:
    """Project camera-space points to ``[N, 2]`` pixels with OpenCV's Brown-Conrady or fisheye model."""
    if fisheye:
        return project_fisheye(points_cam, K, distortion)
    obj = points_cam.reshape(-1, 1, 3).astype(np.float64)
    out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, distortion[:5].reshape(1, 5).astype(np.float64))
    return out.reshape(-1, 2)


def ut_project(
    mean: Float[np.ndarray, "3"],
    cov: Float[np.ndarray, "3 3"],
    K: Float[np.ndarray, "3 3"],
    distortion: Float[np.ndarray, "D"],
    *,
    fisheye: bool,
    centre_fallback: bool = False,
) -> tuple[Float[np.ndarray, "2"], Float[np.ndarray, "2 2"]]:
    """Reference scaled-UT projection; returns ``(mean2d, cov2d)`` in pixels.

    ``centre_fallback`` applies the kernel's rule for a sigma point behind the camera: the centre's undistorted
    pinhole projection stands in for it.
    """
    points = sigma_points(mean, cov)
    projected = project_opencv(points, K, distortion, fisheye=fisheye)
    if centre_fallback:
        behind = points[:, 2] <= 0
        projected[behind] = (K @ (mean / mean[2]))[:2]
    mean2d = (WEIGHTS_MEAN[:, None] * projected).sum(axis=0)
    deltas = projected - mean2d
    cov2d = np.einsum("i,ij,ik->jk", WEIGHTS_COV, deltas, deltas)
    return mean2d, cov2d


def mc_project(
    mean: Float[np.ndarray, "3"],
    cov: Float[np.ndarray, "3 3"],
    K: Float[np.ndarray, "3 3"],
    distortion: Float[np.ndarray, "D"],
    *,
    fisheye: bool,
    n_samples: int = 20_000,
    seed: int = 0,
) -> tuple[Float[np.ndarray, "2"], Float[np.ndarray, "2 2"]]:
    """Monte-Carlo ``(mean2d, cov2d)`` of the projected samples, in pixels."""
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean, cov, size=n_samples)
    projected = project_opencv(samples, K, distortion, fisheye=fisheye)
    return projected.mean(axis=0), np.cov(projected.T)


def conic_from_cov(cov2d: Float[np.ndarray, "2 2"]) -> Float[np.ndarray, "3"]:
    """Inverse 2D covariance packed as the kernel's ``(a, b, c)``."""
    inv = np.linalg.inv(cov2d)
    return np.array([inv[0, 0], inv[0, 1], inv[1, 1]])
