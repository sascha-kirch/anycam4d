/**
 * World-space ray-Gaussian evaluation by whitening with the covariance factor L (Σ = L Lᵀ).
 *
 * D² = min_{t ≥ 0} (p + t d)ᵀΛ(p + t d), Λ = Σ⁻¹, p = o - μ: the ray, not the line. gsplat's
 * ‖grd × gro‖² is the minimum over the whole line, which draws Gaussians reaching behind the camera
 * at forward pixels; here the closest point is clamped to the camera centre instead.
 * p is first moved to the ray's Euclidean closest point to μ: whitening the raw p, which is
 * camera-distance long, loses ~cond(L)·‖p‖/σ_min of float32 precision on thin Gaussians:
 *   y_d = L⁻¹ d,  y_p = L⁻¹ p,  a = y_dᵀy_d,  r = y_p - t·y_d,  D² = ‖r‖²,
 * t = y_dᵀy_p / a, or the shift back to the camera centre when that is smaller.
 *
 * Backward: w = L⁻ᵀ r = Λv (v the whitened offset at the evaluated point), ∂D²/∂μ = -2w, ∂D²/∂L = -2 w rᵀ;
 * the same at the clamped point, where t is fixed rather than optimal.
 * No Σ⁻¹ is ever formed. ∂L/∂opacity = ∂L/∂α · exp(power).
 */

#pragma once

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <optional>
#include <utility>

#include "camera_ops.cuh"
#include "common.h"
#include "linalg.cuh" // triangular solves on the packed factor

// The per-pixel pinhole / fisheye ray now lives in CameraOps<F>::unproject
// (camera_ops.cuh) — the fwd/bwd kernels call it through the family template.

// ─────────────────────────────────────────────────────────────────────────────
// Per-pixel world-space ray
// ─────────────────────────────────────────────────────────────────────────────

/**
 * World-space ray (origin, unit direction) for one pixel; shared by forward and backward,
 * which must derive bit-identical rays.
 *
 * The viewmat may be viewmat_world @ T_denorm, whose rotation block is s·R, so the inverse is
 * (1/s²)·Mᵀ with s read off a column. Returns false outside the camera model's invertible domain;
 * outputs stay finite and the caller skips the pixel.
 */
template <ProjectionFamily F>
__device__ __forceinline__ bool device_pixel_ray(
    const float* __restrict__ vm,
    const float* __restrict__ K,
    const float* __restrict__ dist,
    float px,
    float py,
    float& ox,
    float& oy,
    float& oz,
    float& dx,
    float& dy,
    float& dz
)
{
    const float r00 = vm[0], r01 = vm[1], r02 = vm[2];
    const float r10 = vm[4], r11 = vm[5], r12 = vm[6];
    const float r20 = vm[8], r21 = vm[9], r22 = vm[10];
    const float tx = vm[3], ty = vm[7], tz = vm[11];

    // Camera centre in the Gaussians' frame (see banner: M⁻¹ = (1/s²)Mᵀ).
    const float scale_sq = fmaxf(1e-20f, r00 * r00 + r10 * r10 + r20 * r20);
    const float inv_scale_sq = 1.f / scale_sq;
    ox = -(r00 * tx + r10 * ty + r20 * tz) * inv_scale_sq;
    oy = -(r01 * tx + r11 * ty + r21 * tz) * inv_scale_sq;
    oz = -(r02 * tx + r12 * ty + r22 * tz) * inv_scale_sq;

    const float fx = K[0], fy = K[4], cx = K[2], cy = K[5];
    const float u_n = (px + 0.5f - cx) / fx; // normalised pixel coords
    const float v_n = (py + 0.5f - cy) / fy;

    // Camera-space ray direction (unit) via the family template (pinhole or fisheye).
    float dc_x, dc_y, dc_z;
    const bool ray_valid = CameraOps<F>::unproject(u_n, v_n, dist, dc_x, dc_y, dc_z);
    // Transform to world: d_world = Rᵀ · [dc_x, dc_y, dc_z]
    float dw_x = r00 * dc_x + r10 * dc_y + r20 * dc_z;
    float dw_y = r01 * dc_x + r11 * dc_y + r21 * dc_z;
    float dw_z = r02 * dc_x + r12 * dc_y + r22 * dc_z;
    const float inv_dnorm = rsqrtf(dw_x * dw_x + dw_y * dw_y + dw_z * dw_z);
    dx = dw_x * inv_dnorm;
    dy = dw_y * inv_dnorm;
    dz = dw_z * inv_dnorm;
    return ray_valid;
}

// ─────────────────────────────────────────────────────────────────────────────
// Observation validity — the host mask, bitpacked
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Whether a pixel was observed, from the bitpacked host mask ([C, H, (W + 31) / 32] int32, bit
 * px & 31 of word px >> 5); true when vmask is null. Shared by forward and backward.
 */
__device__ __forceinline__ bool
device_pixel_observed(const int32_t* __restrict__ vmask, uint32_t w_words, uint32_t px, uint32_t py)
{
    if (vmask == nullptr)
        return true;
    const uint32_t word = (uint32_t)vmask[py * w_words + (px >> 5)];
    return ((word >> (px & 31u)) & 1u) != 0u;
}

/**
 * Host side of the bitpacked mask: validate it and hand back (pointer, words per row).
 *
 * A mask sized for a different resolution than the render would gate the wrong pixels
 * silently, so the shape check is loud rather than defensive. An absent mask yields a null
 * pointer — no buffer is materialised, and the kernel takes its unchanged dense path.
 *
 * SHARED by the forward and backward launchers so the two cannot disagree on the layout.
 */
static inline std::pair<const int32_t*, uint32_t>
device_valid_mask_ptr(const std::optional<at::Tensor>& valid_mask, int C, int image_width, int image_height)
{
    const uint32_t w_words = (uint32_t)((image_width + 31) / 32);
    if (!valid_mask.has_value())
        return {nullptr, w_words};
    const at::Tensor& m = valid_mask.value();
    TORCH_CHECK(
        m.dim() == 3 && m.size(0) == C && m.size(1) == image_height && m.size(2) == (int64_t)w_words,
        "valid_mask must be [C, H, (W+31)/32] = [",
        C,
        ", ",
        image_height,
        ", ",
        w_words,
        "]; got a tensor of dim ",
        m.dim()
    );
    TORCH_CHECK(m.scalar_type() == at::kInt, "valid_mask must be int32");
    TORCH_CHECK(m.is_contiguous(), "valid_mask must be contiguous");
    return {m.data_ptr<int32_t>(), w_words};
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared per-batch Gaussian slots
// ─────────────────────────────────────────────────────────────────────────────

/**
 * One batch of Gaussians in shared memory: flat id, (μ, opacity), covariance factor L split 4 + 2. Vector slots so a
 * pixel reads a Gaussian with three loads; GAUSSIAN_BATCH_BYTES per Gaussian.
 */
// Bytes per Gaussian slot, the size both launchers allocate per thread of the block. float4 slots need the
// block size to be a multiple of 4 (16-byte alignment of each slot array after the ids).
constexpr size_t GAUSSIAN_BATCH_BYTES = sizeof(int32_t) + 2 * sizeof(float4) + sizeof(float2);
static_assert(
    (WORLD_RASTERIZE_TILE_SIZE * WORLD_RASTERIZE_TILE_SIZE) % 4 == 0,
    "float4 slots need block_size % 4 == 0"
);

struct GaussianBatch {
    int32_t* ids;
    float4* mu_opac;
    float4* l0123;
    float2* l45;
};

__device__ __forceinline__ GaussianBatch device_gaussian_batch(int* smem, uint32_t block_size)
{
    GaussianBatch b;
    b.ids = (int32_t*)smem;
    b.mu_opac = (float4*)&b.ids[block_size]; // block_size is a multiple of 4: 16-byte aligned
    b.l0123 = &b.mu_opac[block_size];
    b.l45 = (float2*)&b.l0123[block_size];
    return b;
}

/** Fill slot `slot` with Gaussian `gauss_id`. */
__device__ __forceinline__ void device_store_gaussian(
    const GaussianBatch& b,
    uint32_t slot,
    int32_t flat_id,
    int32_t gauss_id,
    const float* __restrict__ means,
    const float* __restrict__ cov_factors,
    const float* __restrict__ opacities
)
{
    const float* L = cov_factors + gauss_id * 6;
    b.ids[slot] = flat_id;
    b.mu_opac[slot] =
        make_float4(means[gauss_id * 3 + 0], means[gauss_id * 3 + 1], means[gauss_id * 3 + 2], opacities[gauss_id]);
    b.l0123[slot] = make_float4(L[0], L[1], L[2], L[3]);
    b.l45[slot] = make_float2(L[4], L[5]);
}

__device__ __forceinline__ void device_load_gaussian(const GaussianBatch& b, uint32_t t, float4& mu_opac, float L[6])
{
    mu_opac = b.mu_opac[t];
    const float4 l0 = b.l0123[t];
    const float2 l1 = b.l45[t];
    L[0] = l0.x;
    L[1] = l0.y;
    L[2] = l0.z;
    L[3] = l0.w;
    L[4] = l1.x;
    L[5] = l1.y;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mahalanobis ray-Gaussian distance
// ─────────────────────────────────────────────────────────────────────────────

/**
 * a = dᵀΛd, the whitened residual r and D² = ‖r‖² (over t ≥ 0) for a unit ray against a Gaussian with packed
 * lower-tri factor L = [l00, l10, l20, l11, l21, l22]; D² >= 0 by construction.
 */
__device__ __forceinline__ void device_ray_gaussian_mahal(
    float ox,
    float oy,
    float oz,
    float dx,
    float dy,
    float dz,
    float mux,
    float muy,
    float muz,
    const float L[6],
    float& a_out,
    float r_out[3],
    float& d2_out
)
{
    // p = o − μ, moved along the unit ray to its closest point to μ
    const float p0[3] = {ox - mux, oy - muy, oz - muz};
    const float s = p0[0] * dx + p0[1] * dy + p0[2] * dz;
    const float p[3] = {p0[0] - s * dx, p0[1] - s * dy, p0[2] - s * dz};
    const float d[3] = {dx, dy, dz};

    float yd[3], yp[3];
    device_solve_lower3x3(L, d, yd);
    device_solve_lower3x3(L, p, yp);

    const float a = yd[0] * yd[0] + yd[1] * yd[1] + yd[2] * yd[2]; // ‖y_d‖² = dᵀΛd
    const float b = yd[0] * yp[0] + yd[1] * yp[1] + yd[2] * yp[2];
    a_out = a;

    // The minimiser lies at ray parameter -s - b/a; behind the camera (< 0) the ray's closest point is o, at shift -s.
    const float t = (a > 0.f) ? fminf(b / a, -s) : 0.f;
    r_out[0] = yp[0] - t * yd[0];
    r_out[1] = yp[1] - t * yd[1];
    r_out[2] = yp[2] - t * yd[2];
    d2_out = r_out[0] * r_out[0] + r_out[1] * r_out[1] + r_out[2] * r_out[2];
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward through Mahalanobis alpha computation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Backward of alpha = opacity · exp(-0.5 D²) to μ, the packed factor L and opacity for one ray, from the forward's
 * whitened residual r. Returns per-ray values; the caller reduces them across the warp. `opacity` is effective.
 */
__device__ __forceinline__ void device_mahal_backward(
    float dL_dalpha,
    float opacity,
    float d2,
    const float r[3],
    const float L[6],
    float dL_dmu[3],
    float dL_dfactor[6],
    float& dL_dopacity
)
{
    const float exp_power = __expf(-0.5f * d2);
    const float dL_dD2 = -0.5f * dL_dalpha * opacity * exp_power;

    // w = L⁻ᵀ r = Λv
    float w[3];
    device_solve_upper3x3(L, r, w);

    // ∂D²/∂μ = -2w
    dL_dmu[0] = -2.f * dL_dD2 * w[0];
    dL_dmu[1] = -2.f * dL_dD2 * w[1];
    dL_dmu[2] = -2.f * dL_dD2 * w[2];

    // r = L⁻¹v at fixed ray parameter (the minimiser, so its own variation drops out): ∂D²/∂L = -2 w rᵀ, lower part.
    const float g = -2.f * dL_dD2;
    dL_dfactor[0] = g * w[0] * r[0];
    dL_dfactor[1] = g * w[1] * r[0];
    dL_dfactor[2] = g * w[2] * r[0];
    dL_dfactor[3] = g * w[1] * r[1];
    dL_dfactor[4] = g * w[2] * r[1];
    dL_dfactor[5] = g * w[2] * r[2];

    dL_dopacity = dL_dalpha * exp_power;
}
