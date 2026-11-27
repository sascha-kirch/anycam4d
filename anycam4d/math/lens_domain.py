"""Where a lens's radial map is physical, its image (the footprint) and its inverse.

A distortion polynomial describes a lens only where its radial map is strictly monotone; past the
first turning point a pixel is not the image of one ray. That is only a bound: a lens's field also
ends at its optics, which no coefficient encodes, so every entry point takes the recorded field
limit ``max_incidence`` (radians). ``None`` asserts the polynomial alone ends the field.

Numpy + cv2, no torch: the torch-free ``data/prep`` package imports it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

from anycam4d.camera.registry import ProjectionFamily


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float

# Relative imaginary part below which a polynomial root counts as real (a double root is ~sqrt(eps) off).
_ROOT_IMAG_REL_TOL = 1e-6

# Bisection on the monotone branch (bracket <= pi, so 40 steps reach ~3e-12 rad); robust at the fold.
_BISECTION_STEPS = 40

# Newton polish after bisection, for polynomials very flat near the fold.
_NEWTON_STEPS = 3

# Samples tracing the boundary image for lenses with tangential/prism terms (no closed form).
_BOUNDARY_SAMPLES = 4096

# Sub-pixel precision of the boundary-polygon rasterisation, as OpenCV's fixed-point shift.
_FILL_SHIFT = 4

#: Pixel index ``i`` is the continuous coordinate ``i + 0.5``, as the world rasterizer samples it.
#: OpenCV treats integers as centres, so coordinates crossing into or out of ``cv2`` shift by 0.5.
PIXEL_CENTRE = 0.5


def first_positive_root(coeffs: Float[np.ndarray, "D"]) -> float:
    """Smallest positive real root of a polynomial in ascending coefficients, ``+inf`` if none."""
    trimmed = np.trim_zeros(np.asarray(coeffs, dtype=np.float64), trim="b")
    if trimmed.size < 2:
        return math.inf
    roots = np.polynomial.polynomial.polyroots(trimmed)
    real = roots[np.abs(roots.imag) <= _ROOT_IMAG_REL_TOL * np.maximum(1.0, np.abs(roots.real))].real
    positive = real[real > 0.0]
    return float(positive.min()) if positive.size else math.inf


def pinhole_radial_polynomials(dist: Float[np.ndarray, "12"]) -> tuple[np.ndarray, np.ndarray]:
    """Numerator and denominator of the Brown-Conrady radial rational in ``s = r²``.

    ``r_d = r · num(s) / den(s)`` with ``num = 1 + k1 s + k2 s² + k3 s³``, ``den = 1 + k4 s + k5 s² + k6 s³``.
    """
    num = np.array([1.0, dist[0], dist[1], dist[4]], dtype=np.float64)
    den = np.array([1.0, dist[5], dist[6], dist[7]], dtype=np.float64)
    return num, den


def pinhole_domain_radius(dist: Float[np.ndarray, "12"]) -> float:
    """Largest undistorted radius where the Brown-Conrady radial map is strictly monotone (``+inf`` if everywhere).

    The first positive root of ``num·den + 2s(num'·den - num·den')`` or of ``den`` (a pole).
    """
    num, den = pinhole_radial_polynomials(dist)
    poly = np.polynomial.polynomial
    d_num = poly.polyder(num)
    d_den = poly.polyder(den)
    derivative_numer = poly.polyadd(
        poly.polymul(num, den),
        poly.polymul([0.0, 2.0], poly.polysub(poly.polymul(d_num, den), poly.polymul(num, d_den))),
    )
    s_max = min(first_positive_root(derivative_numer), first_positive_root(den))
    return math.sqrt(s_max) if math.isfinite(s_max) else math.inf


def fisheye_domain_angle(dist: Float[np.ndarray, "4"]) -> float:
    """Largest incidence angle where ``θ_d(θ)`` is strictly monotone, capped at π.

    The fold is the first positive root of ``1 + 3k1 s + 5k2 s² + 7k3 s³ + 9k4 s⁴`` in ``s = θ²``.
    """
    derivative = np.array([1.0, 3.0 * dist[0], 5.0 * dist[1], 7.0 * dist[2], 9.0 * dist[3]], dtype=np.float64)
    s_max = first_positive_root(derivative)
    theta_max = math.sqrt(s_max) if math.isfinite(s_max) else math.inf
    return min(theta_max, math.pi)


def _fisheye_bound(dist: Float[np.ndarray, "4"], max_incidence: float | None) -> float:
    """The fisheye domain angle, tightened by a recorded field limit when one is known."""
    theta_max = fisheye_domain_angle(dist)
    return theta_max if max_incidence is None else min(theta_max, max_incidence)


def _pinhole_bound(dist: Float[np.ndarray, "12"], max_incidence: float | None) -> float:
    """The pinhole domain radius tightened by ``tan(max_incidence)``; a limit past π/2 constrains nothing."""
    r_max = pinhole_domain_radius(dist)
    if max_incidence is None:
        return r_max
    limit = math.inf if max_incidence >= math.pi / 2 else math.tan(max_incidence)
    return min(r_max, limit)


def fisheye_theta_d(theta: Float[np.ndarray, "..."], dist: Float[np.ndarray, "4"]) -> Float[np.ndarray, "..."]:
    """Kannala-Brandt radial map ``θ_d(θ) = θ (1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)``."""
    s = theta * theta
    return theta * (1.0 + s * (dist[0] + s * (dist[1] + s * (dist[2] + s * dist[3]))))


def _fisheye_theta_d_derivative(
    theta: Float[np.ndarray, "..."],
    dist: Float[np.ndarray, "4"],
) -> Float[np.ndarray, "..."]:
    """``d θ_d/dθ = 1 + 3k1θ² + 5k2θ⁴ + 7k3θ⁶ + 9k4θ⁸`` — strictly positive on the monotone domain."""
    s = theta * theta
    return 1.0 + s * (3.0 * dist[0] + s * (5.0 * dist[1] + s * (7.0 * dist[2] + s * 9.0 * dist[3])))


def fisheye_theta_from_theta_d(
    theta_d: Float[np.ndarray, "..."],
    dist: Float[np.ndarray, "4"],
) -> tuple[Float[np.ndarray, "..."], Bool[np.ndarray, "..."]]:
    """Invert ``θ_d(θ)`` on its monotone branch by bisection plus Newton.

    Returns:
        ``(theta, valid)``; ``theta`` is NaN outside the domain's image.
    """
    theta_max = fisheye_domain_angle(dist)
    theta_d_max = float(fisheye_theta_d(np.asarray(theta_max, dtype=np.float64), dist))

    td = np.asarray(theta_d, dtype=np.float64)
    valid = np.isfinite(td) & (td >= 0.0) & (td <= theta_d_max)

    lo = np.zeros_like(td)
    hi = np.full_like(td, theta_max)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        below = fisheye_theta_d(mid, dist) < td
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    theta = 0.5 * (lo + hi)

    # Clamped to the bracket; skipped where the derivative vanishes at the fold.
    for _ in range(_NEWTON_STEPS):
        slope = _fisheye_theta_d_derivative(theta, dist)
        usable = slope > 0.0
        step = np.where(usable, (fisheye_theta_d(theta, dist) - td) / np.where(usable, slope, 1.0), 0.0)
        theta = np.clip(theta - step, lo, hi)

    return np.where(valid, theta, np.nan), valid


def _distort_normalized(
    x: Float[np.ndarray, "N"],
    y: Float[np.ndarray, "N"],
    dist: Float[np.ndarray, "12"],
) -> tuple[np.ndarray, np.ndarray]:
    """Forward Brown-Conrady map in normalised coordinates; host twin of ``CameraOps<PINHOLE_FAMILY>::project``."""
    r2 = x * x + y * y
    num = 1.0 + r2 * (dist[0] + r2 * (dist[1] + r2 * dist[4]))
    den = 1.0 + r2 * (dist[5] + r2 * (dist[6] + r2 * dist[7]))
    scale = num / den
    a1 = 2.0 * x * y
    xd = x * scale + dist[2] * a1 + dist[3] * (r2 + 2.0 * x * x) + r2 * (dist[8] + r2 * dist[9])
    yd = y * scale + dist[2] * (r2 + 2.0 * y * y) + dist[3] * a1 + r2 * (dist[10] + r2 * dist[11])
    return xd, yd


def _normalized_radius(K: Float[np.ndarray, "3 3"], width: int, height: int) -> Float[np.ndarray, "H W"]:
    """Per-pixel radius in normalised camera coordinates, at the pixel centres the renderer samples."""
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x = (uu + PIXEL_CENTRE - K[0, 2]) / K[0, 0]
    y = (vv + PIXEL_CENTRE - K[1, 2]) / K[1, 1]
    return np.hypot(x, y)


def _disc_mask(K: Float[np.ndarray, "3 3"], width: int, height: int, radius: float) -> Bool[np.ndarray, "H W"]:
    """Mask of pixels whose normalised radius is within ``radius`` (all-``True`` when infinite)."""
    if not math.isfinite(radius):
        return np.ones((height, width), dtype=bool)
    return _normalized_radius(K, width, height) <= radius


def _boundary_fill_mask(
    K: Float[np.ndarray, "3 3"],
    dist: Float[np.ndarray, "12"],
    r_max: float,
    width: int,
    height: int,
) -> Bool[np.ndarray, "H W"]:
    """Footprint of a lens with tangential/prism terms: the polygon traced by the boundary circle's image."""
    if not math.isfinite(r_max):
        return np.ones((height, width), dtype=bool)
    phi = np.linspace(0.0, 2.0 * math.pi, _BOUNDARY_SAMPLES, endpoint=False)
    xd, yd = _distort_normalized(r_max * np.cos(phi), r_max * np.sin(phi), dist)
    # Back to integer pixel indices, which is the grid ``fillPoly`` rasterises on.
    u = K[0, 0] * xd + K[0, 2] - PIXEL_CENTRE
    v = K[1, 1] * yd + K[1, 2] - PIXEL_CENTRE

    scale = float(1 << _FILL_SHIFT)
    polygon = np.stack([np.rint(u * scale), np.rint(v * scale)], axis=-1).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1, shift=_FILL_SHIFT)
    return mask.astype(bool)


def fisheye_domain_radius(dist: Float[np.ndarray, "4"], max_incidence: float | None = None) -> float:
    """Distorted normalised radius of the fisheye rim, tightened by ``max_incidence``."""
    theta_max = _fisheye_bound(dist, max_incidence)
    return float(fisheye_theta_d(np.asarray(theta_max, dtype=np.float64), dist))


def model_footprint(
    K: Float[np.ndarray, "3 3"],
    dist: Float[np.ndarray, "P"],
    family: ProjectionFamily,
    width: int,
    height: int,
    max_incidence: float | None = None,
) -> Bool[np.ndarray, "H W"]:
    """``[H, W]`` mask of pixels imaged by rays inside the lens's monotone domain and recorded field limit.

    Closed form for a purely radial lens, boundary tracing otherwise.
    """
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    if family is ProjectionFamily.FISHEYE_FAMILY:
        return _disc_mask(K, width, height, fisheye_domain_radius(dist, max_incidence))

    r_max = _pinhole_bound(dist, max_incidence)
    if np.any(dist[[2, 3, 8, 9, 10, 11]] != 0.0):
        return _boundary_fill_mask(K, dist, r_max, width, height)
    # Purely radial: the boundary is a circle, so its image is the disc of radius r_d(r_max).
    radius = math.inf
    if math.isfinite(r_max):
        xd, _ = _distort_normalized(np.array([r_max]), np.array([0.0]), dist)
        radius = float(xd[0])
    return _disc_mask(K, width, height, radius)


# OpenCV's default undistort stops after 5 iterations with no epsilon test, leaving pixel-scale
# errors on strongly distorted lenses; converge on epsilon instead.
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-8)


def unproject_pixels(
    pixels: Float[np.ndarray, "N 2"],
    K: Float[np.ndarray, "3 3"],
    dist: Float[np.ndarray, "P"],
    family: ProjectionFamily,
) -> tuple[Float[np.ndarray, "N 3"], Bool[np.ndarray, "N"]]:
    """Unit-sphere rays with signed ``z`` for ``i + 0.5`` pixel coordinates, plus validity.

    Returns:
        ``(dirs [N, 3], valid [N])``; ``dirs`` is NaN where invalid.
    """
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    if family is ProjectionFamily.FISHEYE_FAMILY:
        x_d = (pts[:, 0] - K[0, 2]) / K[0, 0]
        y_d = (pts[:, 1] - K[1, 2]) / K[1, 1]
        theta, valid = fisheye_theta_from_theta_d(np.hypot(x_d, y_d), dist)
        psi = np.arctan2(y_d, x_d)
        sin_theta = np.sin(theta)
        dirs = np.stack([sin_theta * np.cos(psi), sin_theta * np.sin(psi), np.cos(theta)], axis=-1)
        return np.where(valid[:, None], dirs, np.nan), valid

    norm = cv2.undistortPointsIter(pts.reshape(-1, 1, 2), K, dist, None, None, _UNDISTORT_CRITERIA).reshape(-1, 2)
    x, y = norm[:, 0], norm[:, 1]
    valid = np.isfinite(x) & np.isfinite(y) & (np.hypot(x, y) <= pinhole_domain_radius(dist))
    dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    return np.where(valid[:, None], dirs, np.nan), valid


def _footprint_silhouette(mask: Bool[np.ndarray, "H W"]) -> Float[np.ndarray, "M 2"]:
    """Pixel centres on the outline of a footprint (per-row and per-column extremes)."""
    height, width = mask.shape
    rows = np.arange(height)
    cols = np.arange(width)
    col_seen = mask.any(axis=0)
    row_seen = mask.any(axis=1)
    top = mask.argmax(axis=0)
    bottom = height - 1 - mask[::-1].argmax(axis=0)
    left = mask.argmax(axis=1)
    right = width - 1 - mask[:, ::-1].argmax(axis=1)

    pixels = np.concatenate(
        [
            np.stack([cols[col_seen], top[col_seen]], axis=-1),
            np.stack([cols[col_seen], bottom[col_seen]], axis=-1),
            np.stack([left[row_seen], rows[row_seen]], axis=-1),
            np.stack([right[row_seen], rows[row_seen]], axis=-1),
        ]
    ).astype(np.float64)
    return pixels + PIXEL_CENTRE


def max_observed_incidence(
    observed: Bool[np.ndarray, "H W"],
    K: Float[np.ndarray, "3 3"],
    dist: Float[np.ndarray, "P"],
    family: ProjectionFamily,
) -> float | None:
    """Largest incidence among observed pixels (a floor on the field limit), or ``None``; only when none was recorded."""
    if not observed.any():
        return None
    dirs, valid = unproject_pixels(_footprint_silhouette(observed), K, dist, family)
    if not valid.any():
        return None
    dirs = dirs[valid]
    return float(np.arctan2(np.hypot(dirs[:, 0], dirs[:, 1]), dirs[:, 2]).max())
