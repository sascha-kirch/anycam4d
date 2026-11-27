/**
 * common.h — Shared enums and constants for anycam4d CUDA kernels.
 *
 * Exposed to Python via pybind11 (see ext.cpp).
 */

#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Projection family — the compile-time device dispatch tag
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Projection family, selected at compile time; the host switches on it and launches kernel<F>.
 * PINHOLE_FAMILY: pinhole, OpenCV, OpenCV full (12-slot Brown-Conrady). FISHEYE_FAMILY: equidistant and
 * Kannala-Brandt. Mirrors FAMILY_CUDA_ID in anycam4d/camera/registry.py; change both together.
 */
enum ProjectionFamily {
    PINHOLE_FAMILY = 0,
    FISHEYE_FAMILY = 1,
};

// ─────────────────────────────────────────────────────────────────────────────
// Appearance kind — the compile-time colour-leaf dispatch tag
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Temporal colour basis, selected at compile time; the host switches on it and launches kernel<M>.
 * The model is uniform per run (config, never per Gaussian or per step), so the device has no runtime
 * appearance branch — see AppearanceOps<M> in sh4d.cuh. Spatial-only SH is the Additive leaf with zero
 * terms. Mirrors APPEARANCE_CUDA_ID in anycam4d/primitives/appearance.py; change both together.
 */
enum class AppearanceKind {
    Additive = 0,
    Multiplicative = 1,
};

// Highest temporal order the multiplicative basis carries, as published 4DGS fixes it
// (sh_degree_t = 2, train.py:49 — no scene config sets it). The active order is a runtime
// curriculum value; this is the allocated maximum. Mirrors MAX_TEMPORAL_DEGREE in
// anycam4d/primitives/appearance.py.
static constexpr int MAX_TEMPORAL_DEGREE = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Non-owning device camera view (POD, passed by value to templated kernels)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CameraViewDevice — non-owning bundle of the per-batch camera tensors + scalars.
 *
 * Assembled host-side from the flat pybind args (Python guarantees contiguity) and
 * passed by value into each ``kernel<F>`` (lands in parameter registers). The kernels
 * read ``viewmats + c*16``, ``Ks + c*9``, ``distortion + c*CameraOps<F>::n_slots``.
 * There is NO family field — the family is the kernel template parameter, chosen by
 * the host switch. ``dist_stride`` is kept only for the host-side TORCH_CHECK; the
 * kernel uses the ``CameraOps<F>::n_slots`` constexpr for the unrollable access.
 */
struct CameraViewDevice {
    const float* viewmats;   // [C, 4, 4] row-major
    const float* Ks;         // [C, 3, 3] row-major
    const float* distortion; // [C, P] fixed-width (P = dist_stride)
    int C;
    int dist_stride;
    int width;
    int height;
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared rasterization constants
// ─────────────────────────────────────────────────────────────────────────────

#ifndef ALPHA_THRESHOLD
#define ALPHA_THRESHOLD (1.f / 255.f)
#endif

// Per-Gaussian alpha is clamped to this maximum to avoid 1/(1-α) = ∞ in the
// backward compositing recurrence T_i = T_{i+1} / (1 - α_i).
#ifndef MAX_RASTER_ALPHA
#define MAX_RASTER_ALPHA 0.99f
#endif

// Pixel transmittance early-stop threshold for front-to-back compositing.
#ifndef T_RASTER_THRESHOLD
#define T_RASTER_THRESHOLD 1e-4f
#endif

// Tile edge length (square tiles) for the world-space rasterizer.  Must match
// UT_WORLD_RASTERIZE_TILE_SIZE in anycam4d/cuda/ext.py.
#ifndef WORLD_RASTERIZE_TILE_SIZE
#define WORLD_RASTERIZE_TILE_SIZE 16
#endif

// Per-Gaussian density (= exp(-0.5 D²)) cutoff used by the world-space
// rasterizer.  Matches gsplat's MAX_KERNEL_DENSITY_CUTOFF (0.0113 ≈ exp(-4.48),
// i.e. ≈ 3 σ).  Without this cutoff, high-opacity Gaussians whose ray passes
// far from their centre still contribute small but non-zero alpha to every
// pixel they touch, accumulating noise into the front-to-back composite and
// diluting the photometric gradient.
#ifndef MAX_RASTER_KERNEL_DENSITY
#define MAX_RASTER_KERNEL_DENSITY 0.0113f
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Host-side appearance dispatch
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Run one launch statement with ``M`` bound to the kind ``kind`` names, exhaustively.
 *
 * Expands to a switch, so the kind crosses the pybind boundary as an int and becomes a template
 * argument here — never a device branch. Expanded only in translation units that already include
 * torch (it uses TORCH_CHECK for the unknown-kind arm).
 */
#define ANYCAM4D_DISPATCH_APPEARANCE(kind, ...)                                                                        \
    switch (static_cast<AppearanceKind>(kind)) {                                                                       \
    case AppearanceKind::Additive: {                                                                                   \
        constexpr AppearanceKind M = AppearanceKind::Additive;                                                         \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case AppearanceKind::Multiplicative: {                                                                             \
        constexpr AppearanceKind M = AppearanceKind::Multiplicative;                                                   \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    default:                                                                                                           \
        TORCH_CHECK(false, "unsupported appearance kind ", (int)(kind));                                               \
    }

/**
 * Run one launch statement with ``F`` and ``kLens`` bound for the EWA kernels: an undistorted pinhole
 * (``has_lens`` false) takes ``kLens = false`` and today's pinhole device function; every distorted camera
 * takes ``kLens = true`` and its family's lens Jacobian.
 */
#define ANYCAM4D_DISPATCH_EWA_LENS(has_lens, family, ...)                                                              \
    if (!(has_lens)) {                                                                                                 \
        constexpr ProjectionFamily F = PINHOLE_FAMILY;                                                                 \
        constexpr bool kLens = false;                                                                                  \
        __VA_ARGS__;                                                                                                   \
    }                                                                                                                  \
    else {                                                                                                             \
        switch (static_cast<ProjectionFamily>(family)) {                                                               \
        case PINHOLE_FAMILY: {                                                                                         \
            constexpr ProjectionFamily F = PINHOLE_FAMILY;                                                             \
            constexpr bool kLens = true;                                                                               \
            __VA_ARGS__;                                                                                               \
            break;                                                                                                     \
        }                                                                                                              \
        case FISHEYE_FAMILY: {                                                                                         \
            constexpr ProjectionFamily F = FISHEYE_FAMILY;                                                             \
            constexpr bool kLens = true;                                                                               \
            __VA_ARGS__;                                                                                               \
            break;                                                                                                     \
        }                                                                                                              \
        default:                                                                                                       \
            TORCH_CHECK(false, "unsupported projection family ", (int)(family));                                       \
        }                                                                                                              \
    }
