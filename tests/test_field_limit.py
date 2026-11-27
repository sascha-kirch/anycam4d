"""A lens's field is recorded capture metadata, or measured off its mask, not inferred from the polynomial."""

import math

import numpy as np
import pytest
import torch

from anycam4d.camera.registry import ProjectionFamily
from anycam4d.camera.validity import compute_validity
from anycam4d.enums import CameraModelType
from anycam4d.math.lens_domain import max_observed_incidence
from anycam4d.math.lens_domain import model_footprint


# A 170-degree circular lens on an 800x800 frame. Its Kannala-Brandt derivative
# 1 + 3k1 t^2 + 5k2 t^4 has no positive root (9 k1^2 < 20 k2), so the polynomial is monotone all the
# way to theta = pi and says nothing whatsoever about where the glass stops.
_CIRCULAR_K = np.array([[364.15828292367513, 0.0, 400.0], [0.0, 364.15828292367513, 400.0], [0.0, 0.0, 1.0]])
_CIRCULAR_DIST = (-0.25, 0.06, 0.0, 0.0)
_CIRCULAR_WH = (800, 800)
_CIRCULAR_LIMIT_DEG = 85.0

# An equidistant lens (k1..k4 = 0) is monotone everywhere by construction, so the recorded limit is
# the only thing that can bound it.
_EQUIDISTANT_DIST = np.zeros(4)


def test_validity_excludes_the_frame_corners_only_with_the_limit() -> None:
    """Without the limit the polynomial claims the whole frame, corners included."""
    K = torch.from_numpy(_CIRCULAR_K.astype(np.float32))
    dist = torch.tensor([*_CIRCULAR_DIST], dtype=torch.float32)
    without = compute_validity(CameraModelType.FISHEYE_OPENCV, K, dist, *_CIRCULAR_WH)
    with_limit = compute_validity(
        CameraModelType.FISHEYE_OPENCV, K, dist, *_CIRCULAR_WH, max_incidence_deg=_CIRCULAR_LIMIT_DEG
    )
    assert bool(without.all())
    assert not bool(with_limit[0, 0])
    assert bool(with_limit[400, 400])
    # The image circle inscribed in a square frame is pi/4 of it.
    assert with_limit.float().mean().item() == pytest.approx(math.pi / 4, abs=0.01)
    assert bool((with_limit & ~without).sum() == 0)


def test_the_measurement_recovers_the_limit_it_was_rendered_from() -> None:
    """A floor, never an overstatement: the outermost observed pixel centre is inside the true rim."""
    K = np.array([[200.0, 0.0, 400.0], [0.0, 200.0, 400.0], [0.0, 0.0, 1.0]])
    family = ProjectionFamily.FISHEYE_FAMILY
    observed = model_footprint(K, _EQUIDISTANT_DIST, family, 800, 800, math.radians(70.0))
    measured = max_observed_incidence(observed, K, _EQUIDISTANT_DIST, family)
    assert math.degrees(measured) == pytest.approx(70.0, abs=0.02)
    assert math.degrees(measured) <= 70.0


def test_the_measurement_reads_a_brown_conrady_lens() -> None:
    """A pinhole-family lens is measured on the canonical 12-slot vector, not a compact tuple."""
    K = np.array([[533.3, 0.0, 400.0], [0.0, 533.3, 300.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(12)
    observed = np.ones((600, 800), dtype=bool)
    measured = max_observed_incidence(observed, K, dist, ProjectionFamily.PINHOLE_FAMILY)
    corner = math.degrees(math.atan(math.hypot(399.5, 299.5) / 533.3))
    assert math.degrees(measured) == pytest.approx(corner, abs=1e-6)
