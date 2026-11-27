"""Render capability gate: unbuildable render configs fail at construction."""

from __future__ import annotations

import pytest

from anycam4d.config.rendering import RenderingConfig
from anycam4d.enums import BackendType
from anycam4d.enums import RasterizeMode
from anycam4d.renderer.render_registry import check_render_capability


@pytest.mark.parametrize("backend", [BackendType.ANYCAM4D])
def test_ray_backends_reject_antialiased(backend: BackendType) -> None:
    """Ray backends reject ``antialiased``: the 2D compensation does not describe what they integrate."""
    check_render_capability(backend, rasterize_mode=RasterizeMode.CLASSIC)

    with pytest.raises(ValueError, match="does not support rasterize_mode=antialiased"):
        check_render_capability(backend, rasterize_mode=RasterizeMode.ANTIALIASED)


def test_ewa_supports_antialiased() -> None:
    """EWA integrates the same 2D covariance it blurs, so the compensation is the right quantity."""
    check_render_capability(BackendType.EWA, rasterize_mode=RasterizeMode.ANTIALIASED)


def test_factory_rejects_ut_antialiased() -> None:
    """The factory runs the gate: a UT config requesting antialiased is rejected at construction."""
    from anycam4d.renderer.factory import BackendFactory

    with pytest.raises(ValueError, match="does not support rasterize_mode=antialiased"):
        BackendFactory.create(
            RenderingConfig(
                backend=BackendType.ANYCAM4D,
                packed=False,
                rasterize_mode=RasterizeMode.ANTIALIASED,
            )
        )
