"""Bindings of the JIT-built CUDA extension (list in ``cuda/csrc/ext.cpp``).

The build runs on first import and is cached per GPU architecture under ``TORCH_EXTENSIONS_DIR`` (default
``~/.cache/torch_extensions/``). A failed build leaves ``HAS_CUDA_EXT`` false and the cause in ``BUILD_ERROR``;
``pixi run cuda-build`` shows the compiler output.
"""

from __future__ import annotations


try:
    from anycam4d.cuda.build import build_cuda_ext as _build

    _ext = _build()
    launch_ewa_4d_fused_conditioned_projection_fwd_kernel = _ext.launch_ewa_4d_fused_conditioned_projection_fwd_kernel
    launch_ewa_4d_fused_conditioned_projection_bwd_kernel = _ext.launch_ewa_4d_fused_conditioned_projection_bwd_kernel
    launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel = (
        _ext.launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel
    )
    launch_ewa_4d_fused_packed_conditioned_projection_bwd_kernel = (
        _ext.launch_ewa_4d_fused_packed_conditioned_projection_bwd_kernel
    )
    ut_4d_fused_projection_fwd = _ext.ut_4d_fused_projection_fwd
    ut_4d_fused_projection_bwd = _ext.ut_4d_fused_projection_bwd
    ut_4d_fused_packed_projection_fwd = _ext.ut_4d_fused_packed_projection_fwd
    ut_4d_fused_packed_projection_bwd = _ext.ut_4d_fused_packed_projection_bwd
    rasterize_from_world_fwd = _ext.rasterize_from_world_fwd
    rasterize_from_world_bwd = _ext.rasterize_from_world_bwd
    ut_4d_rolling_shutter_projection_fwd = _ext.ut_4d_rolling_shutter_projection_fwd
    rasterize_from_world_rolling_shutter_fwd = _ext.rasterize_from_world_rolling_shutter_fwd
    ut_4d_depth_of_field_projection_fwd = _ext.ut_4d_depth_of_field_projection_fwd
    ut_4d_dispersion_projection_fwd = _ext.ut_4d_dispersion_projection_fwd
    rasterize_from_world_dispersion_fwd = _ext.rasterize_from_world_dispersion_fwd
    stso_adam_update = _ext.stso_adam_update
    ProjectionFamily = _ext.ProjectionFamily
    AppearanceKind = _ext.AppearanceKind
    MAX_TEMPORAL_DEGREE_CUDA: int = _ext.MAX_TEMPORAL_DEGREE
    HAS_CUDA_EXT: bool = True
    BUILD_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001  (ImportError, OSError, RuntimeError from nvcc)
    HAS_CUDA_EXT = False
    BUILD_ERROR = exc
    launch_ewa_4d_fused_conditioned_projection_fwd_kernel = None  # type: ignore[assignment]
    launch_ewa_4d_fused_conditioned_projection_bwd_kernel = None  # type: ignore[assignment]
    launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel = None  # type: ignore[assignment]
    launch_ewa_4d_fused_packed_conditioned_projection_bwd_kernel = None  # type: ignore[assignment]
    ut_4d_fused_projection_fwd = None  # type: ignore[assignment]
    ut_4d_fused_projection_bwd = None  # type: ignore[assignment]
    ut_4d_fused_packed_projection_fwd = None  # type: ignore[assignment]
    ut_4d_fused_packed_projection_bwd = None  # type: ignore[assignment]
    rasterize_from_world_fwd = None  # type: ignore[assignment]
    rasterize_from_world_bwd = None  # type: ignore[assignment]
    ut_4d_rolling_shutter_projection_fwd = None  # type: ignore[assignment]
    rasterize_from_world_rolling_shutter_fwd = None  # type: ignore[assignment]
    ut_4d_depth_of_field_projection_fwd = None  # type: ignore[assignment]
    ut_4d_dispersion_projection_fwd = None  # type: ignore[assignment]
    rasterize_from_world_dispersion_fwd = None  # type: ignore[assignment]
    stso_adam_update = None  # type: ignore[assignment]
    ProjectionFamily = None  # type: ignore[assignment]
    AppearanceKind = None  # type: ignore[assignment]
    MAX_TEMPORAL_DEGREE_CUDA = None  # type: ignore[assignment]

# Tile size used by the world-space rasterization kernels (rasterize_from_world_fwd/bwd).
# Must match WORLD_RASTERIZE_TILE_SIZE in anycam4d/cuda/csrc/common.h.
UT_WORLD_RASTERIZE_TILE_SIZE: int = 16
