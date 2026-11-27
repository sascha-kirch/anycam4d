"""Bound how many pixels the iterative ray solve moves across the central/peripheral boundary.

Rays from ``camera_rays`` are refined by Gauss-Newton against OpenCV's analytic forward model to
machine precision; region flips are counted inside the observed region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.camera.registry import ProjectionFamily
from anycam4d.camera.validity import compute_validity
from anycam4d.enums import CameraModelType
from anycam4d.math.camera_rays import camera_rays
from anycam4d.math.lens_domain import PIXEL_CENTRE
from anycam4d.math.region import PinholeField
from anycam4d.math.region import central_mask_from_rays


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor


@dataclass(frozen=True)
class LensSpec:
    """A camera model + intrinsics, independent of any dataset on disk."""

    name: str
    model: CameraModelType
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = ()

    def intrinsics(self) -> np.ndarray:
        """The ``[3, 3]`` intrinsic matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def dist_vector(self) -> np.ndarray:
        """The canonical fixed-width, zero-filled distortion vector for this model's family."""
        n = 4 if CAMERA_REGISTRY[self.model].family is ProjectionFamily.FISHEYE_FAMILY else 12
        d = np.zeros(n)
        d[: len(self.distortion)] = self.distortion
        return d


# The three dynopsis scene-1 lenses, verbatim from their `train_dataset.json` camera blocks. Kept
# here so the bound can be reproduced without the dataset on disk.
DYNOPSIS_SCENE1_LENSES: tuple[LensSpec, ...] = (
    LensSpec("pinhole", CameraModelType.PINHOLE, 533.3333333333334, 533.3333333333334, 400.0, 300.0, 800, 600),
    LensSpec(
        "opencv",
        CameraModelType.OPENCV,
        533.3333333333334,
        533.3333333333334,
        400.0,
        300.0,
        800,
        600,
        (-0.28, 0.06, 0.0, 0.0, 0.0),
    ),
    LensSpec(
        "fisheye_kb",
        CameraModelType.FISHEYE_OPENCV,
        364.15828292367513,
        364.15828292367513,
        400.0,
        400.0,
        800,
        800,
        (-0.25, 0.06, 0.0, 0.0),
    ),
)


def project_points(rays: np.ndarray, K: np.ndarray, dist: np.ndarray, model: CameraModelType) -> np.ndarray:
    """Project ``z == 1`` camera-space rays to ``[N, 2]`` pixels with the analytic lens model."""
    obj = np.ascontiguousarray(rays, dtype=np.float64).reshape(-1, 1, 3)
    zero = np.zeros((3, 1))
    family = CAMERA_REGISTRY[model].family
    if family is ProjectionFamily.FISHEYE_FAMILY:
        pix, _ = cv2.fisheye.projectPoints(obj, zero, zero, K, dist)
    else:
        pix, _ = cv2.projectPoints(obj, zero, zero, K, dist)
    return pix.reshape(-1, 2)


def refine_rays(
    rays: Float[Tensor, "H W 3"],
    K: np.ndarray,
    dist: np.ndarray,
    model: CameraModelType,
    targets: np.ndarray,
    *,
    iters: int = 12,
) -> np.ndarray:
    """Gauss-Newton-refine rays until each projects onto its target pixel; returns ``[N, 3]``."""
    xy = rays.reshape(-1, 3).double().numpy()[:, :2].copy()
    eps = 1e-7
    for _ in range(iters):
        base = project_points(np.concatenate([xy, np.ones((len(xy), 1))], axis=1), K, dist, model)
        residual = base - targets
        jac = np.empty((len(xy), 2, 2))
        for axis in range(2):
            step = xy.copy()
            step[:, axis] += eps
            moved = project_points(np.concatenate([step, np.ones((len(xy), 1))], axis=1), K, dist, model)
            jac[:, :, axis] = (moved - base) / eps
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        ok = np.abs(det) > 1e-12
        delta = np.zeros_like(xy)
        inv = np.empty_like(jac)
        inv[:, 0, 0], inv[:, 1, 1] = jac[:, 1, 1], jac[:, 0, 0]
        inv[:, 0, 1], inv[:, 1, 0] = -jac[:, 0, 1], -jac[:, 1, 0]
        delta[ok] = np.einsum("nij,nj->ni", inv[ok], residual[ok]) / det[ok, None]
        xy -= delta
    return np.concatenate([xy, np.ones((len(xy), 1))], axis=1)


@dataclass(frozen=True)
class BoundaryBound:
    """Region-membership error of one camera's ray field, in pixels and in flips."""

    model: CameraModelType
    width: int
    height: int
    observed_px: int
    central_px: int
    roundtrip_median_px: float
    roundtrip_p99_px: float
    roundtrip_max_px: float
    unsolved_px: int  # observed pixels whose round trip misses by > 1 px
    unsolved_min_angle_deg: float  # smallest field angle among them (nan if none)
    unsolved_all_peripheral: bool  # every unsolved pixel is peripheral under both ray fields
    solver_flips: int

    @property
    def solver_flip_fraction(self) -> float:
        """Flips from the solver residual, as a fraction of the observed pixels."""
        return self.solver_flips / self.observed_px


def measure_boundary(
    K: np.ndarray,
    dist: np.ndarray,
    model: CameraModelType,
    width: int,
    height: int,
) -> BoundaryBound:
    """Bound the region boundary's instrument error for one camera."""
    K_t = torch.tensor(K, dtype=torch.float32)
    d_t = torch.tensor(dist, dtype=torch.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])

    rays, ray_valid = camera_rays(K_t, d_t, model, width, height)
    observed: Bool[Tensor, "h w"] = compute_validity(model, K_t, d_t, width, height) & ray_valid

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    targets = np.stack([uu.ravel(), vv.ravel()], axis=-1) + PIXEL_CENTRE
    refined = torch.from_numpy(refine_rays(rays, K, dist, model, targets)).float().reshape(height, width, 3)

    err_full = np.linalg.norm(project_points(rays.reshape(-1, 3).double().numpy(), K, dist, model) - targets, axis=-1)
    obs_flat = observed.reshape(-1).numpy()
    err = err_full[obs_flat]

    field = PinholeField(fx, fy, float(K[0, 2]), float(K[1, 2]), float(width), float(height))
    central_solver = central_mask_from_rays(rays, field)
    central_ref = central_mask_from_rays(refined, field)

    # Rim pixels outside the analytic model's invertible domain: their field angle says whether any is near the boundary.
    unsolved = (err_full > 1.0) & obs_flat
    angle = np.degrees(
        np.arccos(
            np.clip(
                (rays[..., 2] / rays.norm(dim=-1).clamp(min=1e-12)).reshape(-1).numpy(),
                -1.0,
                1.0,
            )
        )
    )
    peripheral_both = (~central_solver.reshape(-1).numpy()) & (~central_ref.reshape(-1).numpy())

    return BoundaryBound(
        model=model,
        width=width,
        height=height,
        observed_px=int(observed.sum()),
        central_px=int((central_solver & observed).sum()),
        roundtrip_median_px=float(np.median(err)),
        roundtrip_p99_px=float(np.quantile(err, 0.99)),
        roundtrip_max_px=float(err.max()),
        unsolved_px=int(unsolved.sum()),
        unsolved_min_angle_deg=float(angle[unsolved].min()) if unsolved.any() else float("nan"),
        unsolved_all_peripheral=bool(peripheral_both[unsolved].all()) if unsolved.any() else True,
        solver_flips=int(((central_solver != central_ref) & observed).sum()),
    )


def measure_lens(lens: LensSpec) -> BoundaryBound:
    """Bound the region boundary's instrument error for a ``LensSpec``."""
    return measure_boundary(lens.intrinsics(), lens.dist_vector(), lens.model, lens.width, lens.height)


def main() -> None:
    """Print the bound for each dynopsis scene-1 lens."""
    for lens in DYNOPSIS_SCENE1_LENSES:
        b = measure_lens(lens)
        print(  # noqa: T201 — this is a CLI report
            f"{lens.name:11s} {b.width}x{b.height} observed={b.observed_px} "
            f"central={b.central_px} ({b.central_px / b.observed_px * 100:.1f}%) | "
            f"round trip med={b.roundtrip_median_px:.2e} p99={b.roundtrip_p99_px:.2e} "
            f"max={b.roundtrip_max_px:.2e} px | unsolved>1px={b.unsolved_px} "
            f"(min field angle {b.unsolved_min_angle_deg:.1f} deg, all peripheral={b.unsolved_all_peripheral}) | "
            f"solver flips={b.solver_flips} ({b.solver_flip_fraction * 100:.4f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
