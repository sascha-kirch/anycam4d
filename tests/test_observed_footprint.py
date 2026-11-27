"""A lens observes the image of its physical domain; a circular lens observes a disc, not its frame."""

import math

import numpy as np
import pytest

from anycam4d.camera.registry import ProjectionFamily
from anycam4d.math.lens_domain import fisheye_domain_angle
from anycam4d.math.lens_domain import model_footprint
from anycam4d.math.lens_domain import unproject_pixels


# A circular Kannala-Brandt fisheye: k1 = -0.3 folds the radial map at 60.4 deg, whose image sits at
# 70.3 px — an image circle comfortably inside the 200x200 frame, with the corners never imaged.
_CIRCULAR_K = np.array([[100.0, 0.0, 99.5], [0.0, 100.0, 99.5], [0.0, 0.0, 1.0]])
_CIRCULAR_DIST = np.array([-0.3, 0.0, 0.0, 0.0])
_CIRCULAR_WH = (200, 200)

# A lens that genuinely fills its frame: the polynomial folds well outside the frame's own corner.
_FULL_FRAME_K = np.array([[533.3, 0.0, 399.5], [0.0, 533.3, 299.5], [0.0, 0.0, 1.0]])
_FULL_FRAME_DIST = np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_FULL_FRAME_WH = (800, 600)

# A 170-degree circular lens whose Kannala-Brandt derivative has no positive root, so the polynomial
# is monotone to pi and reports the whole frame — including corners at 105 degrees incidence that the
# lens never imaged.
_UNBOUNDED_K = np.array([[364.2, 0.0, 399.5], [0.0, 364.2, 399.5], [0.0, 0.0, 1.0]])
_UNBOUNDED_DIST = np.array([-0.25, 0.06, 0.0, 0.0])


def test_the_image_circle_is_the_footprint_not_the_frame() -> None:
    """The fixture really is circular: the frame's corners are outside the lens's domain."""
    footprint = model_footprint(_CIRCULAR_K, _CIRCULAR_DIST, ProjectionFamily.FISHEYE_FAMILY, *_CIRCULAR_WH)

    assert 0.3 < footprint.mean() < 0.45  # a disc of radius 70.3 px in a 200x200 frame
    assert not footprint[0, 0] and not footprint[-1, -1]
    assert footprint[99, 99]  # the optical axis is observed


def test_the_border_ring_of_a_circular_lens_has_no_rays() -> None:
    """Past the fold the polynomial is not injective, so the frame's border unprojects to nothing."""
    assert math.degrees(fisheye_domain_angle(_CIRCULAR_DIST)) == pytest.approx(60.4, abs=0.1)

    width, height = _CIRCULAR_WH
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    ring = np.concatenate(
        [
            np.stack([cols, np.full(width, 0.5)], axis=-1),
            np.stack([cols, np.full(width, height - 0.5)], axis=-1),
            np.stack([np.full(height, 0.5), rows], axis=-1),
            np.stack([np.full(height, width - 0.5), rows], axis=-1),
        ]
    )
    dirs, valid = unproject_pixels(ring, _CIRCULAR_K, _CIRCULAR_DIST, ProjectionFamily.FISHEYE_FAMILY)
    assert not valid.any()
    assert np.isnan(dirs).all()


def test_a_lens_that_fills_its_frame_observes_all_of_it() -> None:
    footprint = model_footprint(_FULL_FRAME_K, _FULL_FRAME_DIST, ProjectionFamily.PINHOLE_FAMILY, *_FULL_FRAME_WH)
    assert footprint.all()


def test_a_monotone_polynomial_is_not_a_statement_about_the_field() -> None:
    """Invertible everywhere, yet the frame corners were never imaged: only a recorded limit says so."""
    family = ProjectionFamily.FISHEYE_FAMILY
    assert model_footprint(_UNBOUNDED_K, _UNBOUNDED_DIST, family, 800, 800).all()

    bounded = model_footprint(_UNBOUNDED_K, _UNBOUNDED_DIST, family, 800, 800, math.radians(85.0))
    assert not bounded[0, 0]
    assert bounded[400, 400]
