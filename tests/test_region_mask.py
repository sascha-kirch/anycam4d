"""CPU unit tests for the central/peripheral eval-region masks (math/region.py)."""

from __future__ import annotations

import numpy as np
import torch

from anycam4d.camera.registry import ProjectionFamily
from anycam4d.math.lens_domain import unproject_pixels
from anycam4d.math.region import build_region_masks
from anycam4d.math.region import PinholeField
from anycam4d.math.region import central_mask_from_rays


def _field(fx: float, fy: float, width: float, height: float) -> PinholeField:
    """Reference pinhole with its principal point at the image centre."""
    return PinholeField(fx, fy, width / 2.0, height / 2.0, width, height)


def _separable_rays(out_w: int, out_h: int, fov_x: float, fov_y: float) -> torch.Tensor:
    """Synthetic camera-space rays whose x-angle depends on the column only and y-angle on the row only."""
    fx_b = out_w / fov_x
    fy_b = out_h / fov_y
    cx_b = (out_w - 1) / 2.0
    cy_b = (out_h - 1) / 2.0
    px = torch.arange(out_w, dtype=torch.float64)
    py = torch.arange(out_h, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    theta = (grid_x + 0.5 - cx_b) / fx_b
    phi = (grid_y + 0.5 - cy_b) / fy_b
    dirs = torch.stack([torch.tan(theta), torch.tan(phi), torch.ones_like(theta)], dim=-1)
    return dirs / dirs.norm(dim=-1, keepdim=True)


def test_central_mask_is_rectangle_for_separable_rays() -> None:
    """Rays separable in (x, y) -> the central boundary is an axis-aligned rectangle."""
    out_w, out_h = 64, 48
    fov_x, fov_y = 0.8, 0.6
    rays = _separable_rays(out_w, out_h, fov_x, fov_y)

    # Reference half-FoV narrower than the render FoV -> a peripheral ring exists.
    fx_ref, fy_ref = 200.0, 200.0
    w_ref, h_ref = 96.0, 72.0
    central = central_mask_from_rays(rays, _field(fx_ref, fy_ref, w_ref, h_ref))

    assert central.any() and (~central).any(), "test setup must produce both a central and peripheral region"
    # Rectangle == a Cartesian product of a row set and a column set (theta indep. of y, phi
    # indep. of x): reconstructing it from row/column marginals must reproduce the mask exactly.
    row_has_any = central.any(dim=1)
    col_has_any = central.any(dim=0)
    assert torch.equal(central, row_has_any.unsqueeze(1) & col_has_any.unsqueeze(0)), (
        "separable central mask must be a perfect axis-aligned rectangle"
    )


def test_central_mask_is_curved_in_fisheye_space() -> None:
    """KB fisheye rays couple x and y through a shared radial term -> the boundary is not a rectangle."""
    out_w, out_h = 64, 48
    fx, fy = 30.0, 30.0
    K = torch.tensor([[fx, 0.0, (out_w - 1) / 2.0], [0.0, fy, (out_h - 1) / 2.0], [0.0, 0.0, 1.0]])
    kb = np.array([0.1, 0.01, 0.001, 0.02])

    uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float64), np.arange(out_h, dtype=np.float64))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=-1)
    dirs, _ = unproject_pixels(pts, K.numpy(), kb, ProjectionFamily.FISHEYE_FAMILY)
    rays = torch.from_numpy(dirs.reshape(out_h, out_w, 3))

    # Reference half-FoV narrow enough to carve out a genuine peripheral ring.
    central = central_mask_from_rays(rays, _field(60.0, 60.0, out_w, out_h))
    assert central.any() and (~central).any(), "test setup must produce both a central and peripheral region"

    # Same Cartesian-product reconstruction as the separable test: KB couples the two axes through a
    # shared radial distortion term, so the mask is not a product of independent row/column sets.
    row_has_any = central.any(dim=1)
    col_has_any = central.any(dim=0)
    assert not torch.equal(central, row_has_any.unsqueeze(1) & col_has_any.unsqueeze(0)), (
        "KB fisheye central mask must not be a perfect axis-aligned rectangle"
    )


def test_central_mask_degenerate_for_matching_pinhole_reference() -> None:
    """Reference FoV equal to the render camera's own FoV -> central == full frame (degenerate)."""
    out_w, out_h = 32, 24
    fx, fy = 50.0, 50.0
    py, px = torch.meshgrid(
        torch.arange(out_h, dtype=torch.float64), torch.arange(out_w, dtype=torch.float64), indexing="ij"
    )
    x = (px - (out_w - 1) / 2.0) / fx
    y = (py - (out_h - 1) / 2.0) / fy
    dirs = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    rays = dirs / dirs.norm(dim=-1, keepdim=True)

    central = central_mask_from_rays(rays, _field(fx, fy, out_w, out_h))
    assert central.all(), "central box == full frame when the reference FoV matches the render camera's own FoV"


def test_build_region_masks_peripheral_is_complement_and_ands_valid() -> None:
    out_w, out_h = 64, 48
    rays = _separable_rays(out_w, out_h, 0.8, 0.6)
    valid = torch.ones(out_h, out_w, dtype=torch.bool)
    valid[:, : out_w // 4] = False  # a masked-out strip, e.g. a border validity mask

    masks = build_region_masks(rays, _field(200.0, 200.0, 96.0, 72.0), valid=valid)
    central, peripheral, v = masks["central"], masks["peripheral"], masks["valid"]

    assert torch.equal(v, valid)
    assert torch.equal(peripheral, (~central_mask_from_rays(rays, _field(200.0, 200.0, 96.0, 72.0))) & valid)
    assert not (central & peripheral).any(), "central and peripheral must be disjoint"
    assert not (central & ~valid).any(), "central must respect the valid mask"


def test_build_region_masks_defaults_valid_to_all_true() -> None:
    rays = _separable_rays(16, 16, 0.5, 0.5)
    masks = build_region_masks(rays, _field(200.0, 200.0, 96.0, 72.0))
    assert masks["valid"].all()
