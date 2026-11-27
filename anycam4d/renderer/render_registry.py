"""The single render-backend capability gate.

``rasterize_mode`` is the one render axis not every backend supports: whether the projection derives
a Mip-Splatting opacity compensation from the 2D covariance blur. That is only meaningful where the
rendered alpha comes from that same 2D covariance — the EWA rasterizer. The ray backend (UT) projects
only to build the tile acceleration structure and evaluates alpha against the 3D covariance along
the ray, which a 2D determinant ratio does not describe, so ``antialiased`` is rejected there rather
than silently ignored. ``packed`` is not gated: both backends build it and it renders identically.

``check_render_capability`` is called once at backend construction (``BackendFactory.create``) so an
unbuildable combination fails fast with a clear message rather than silently mis-rendering.
"""

from __future__ import annotations

from anycam4d.enums import BackendType
from anycam4d.enums import RasterizeMode
from anycam4d.enums import SpatialTrigger


# The rasterize modes each backend implements. Mirrors the conf/rendering group files.
RASTERIZE_MODES: dict[BackendType, frozenset[RasterizeMode]] = {
    BackendType.EWA: frozenset({RasterizeMode.CLASSIC, RasterizeMode.ANTIALIASED}),
    BackendType.ANYCAM4D: frozenset({RasterizeMode.CLASSIC}),
}


def check_render_capability(backend: BackendType, rasterize_mode: RasterizeMode = RasterizeMode.CLASSIC) -> None:
    """Reject a render configuration the backend does not build.

    Args:
        backend: Active render backend.
        rasterize_mode: Selected rasterization mode.

    Raises:
        ValueError: If the backend is unknown or does not implement the rasterization mode.
    """
    supported_modes = RASTERIZE_MODES.get(backend)
    if supported_modes is None:
        raise ValueError(f"Unknown backend {backend.value}: no render capability declared.")
    if rasterize_mode not in supported_modes:
        supported = sorted(m.value for m in supported_modes)
        raise ValueError(
            f"{backend.value} backend does not support rasterize_mode={rasterize_mode.value}; "
            f"supported: {supported}. The Mip-Splatting compensation is derived from the 2D "
            f"covariance blur, which on a ray backend only sizes the tile bound — the rendered "
            f"alpha comes from the 3D covariance along the ray and is unrelated to it."
        )


# Backends whose projection differentiates a screen-space ``means2d`` — the only ones whose
# ``means2d.grad`` is a usable densification signal. The ray backend (UT forward-only tile
# association) emits a ``means2d`` that carries no gradient, so the screen-space
# spatial trigger silently accumulates nothing there and must be rejected.
SCREENSPACE_TRIGGER_BACKENDS: frozenset[BackendType] = frozenset({BackendType.EWA})


def check_spatial_trigger_capability(backend: BackendType, spatial_trigger: SpatialTrigger) -> None:
    """Reject a screen-space spatial trigger on a backend that cannot differentiate ``means2d``.

    The parameter-space trigger (``|∂L/∂means|``) works on every backend and is never rejected;
    only ``screenspace`` is gated, to the ``means2d``-differentiating backends.

    Args:
        backend: Active render backend.
        spatial_trigger: Requested spatial densification trigger source.

    Raises:
        ValueError: If ``spatial_trigger == SCREENSPACE`` and the backend does not emit a
            differentiable screen-space ``means2d``.
    """
    if spatial_trigger == SpatialTrigger.SCREENSPACE and backend not in SCREENSPACE_TRIGGER_BACKENDS:
        raise ValueError(
            f"spatial_trigger=screenspace requires a backend with a differentiable screen-space "
            f"means2d gradient (one of {sorted(b.value for b in SCREENSPACE_TRIGGER_BACKENDS)}); "
            f"the {backend.value} backend has none. Use spatial_trigger=paramspace on the "
            f"{backend.value} backend, or switch to the EWA backend."
        )
