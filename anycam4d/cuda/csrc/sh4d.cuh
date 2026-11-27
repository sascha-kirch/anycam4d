/**
 * Spatial SH (up to degree 3) and additive 4D SH colour, adapted from gsplat's sh_coeffs_to_color_fast
 * (Copyright the Regents of the University of California, Nerfstudio Team and contributors; Apache License 2.0).
 * Coefficients: ShCoeffs, DC and k ≥ 1 on separate pointers, for K = (degree + 1)² bases. Temporal: T = 2 ·
 * n_temporal_freqs interleaved [cos(π f dir_t), sin(π f dir_t)] with dir_t = tau - mu_t. The +0.5 bias and clamp are
 * applied once in device_eval_sh4d_rgb.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h" // AppearanceKind — the compile-time colour-leaf dispatch tag

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ============================================================================
// Named SH polynomial constants
// See: Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013.
// https://jcgt.org/published/0002/02/06/
// ============================================================================

// degree 0
static constexpr float SH_C0 = 0.2820947917738781f; // 1/(2√π)

// degree 1
static constexpr float SH_C1 = 0.48860251190292f; // √(3/(4π))

// degree 2
static constexpr float SH_C2_0 = 0.5462742152960395f;  // (1/2)√(15/π)
static constexpr float SH_C2_1 = 1.092548430592079f;   // √(15/π)   (used negated below)
static constexpr float SH_C2_2 = 0.9461746957575601f;  // (1/4)√(5/π) * 3  (leading coeff of P2_0)
static constexpr float SH_C2_2B = 0.3153915652525201f; // (1/4)√(5/π)      (constant term of P2_0)

// degree 3
static constexpr float SH_C3_0 = 0.5900435899266435f;  // (1/4)√(35/(2π))
static constexpr float SH_C3_1 = 1.445305721320277f;   // (1/2)√(105/π)    (leading)
static constexpr float SH_C3_2 = 2.285228997322329f;   // (1/4)√(105/π) * 4
static constexpr float SH_C3_2B = 0.4570457994644658f; // (1/4)√(105/π)
static constexpr float SH_C3_3 = 1.865881662950577f;   // ...
static constexpr float SH_C3_3B = 1.119528997770346f;  // ...

// ShCoeffs / ShGrad

/**
 * One Gaussian's SH block: DC from ``dc[c]``, basis k ≥ 1 from ``hi[(k - 1) * 3 + c]``. Splitting the
 * pointers lets the kernels read the sh0 and shN parameters in place, with no concatenated copy.
 */
struct ShCoeffs {
    const float* __restrict__ dc;
    const float* __restrict__ hi;

    __device__ __forceinline__ float at(int k, int c) const
    {
        return k == 0 ? dc[c] : hi[(k - 1) * 3 + c];
    }

    // A block stored contiguously, DC first (the multiplicative temporal blocks).
    __device__ __forceinline__ static ShCoeffs contiguous(const float* p)
    {
        return {p, p + 3};
    }
};

struct ShGrad {
    float* dc;
    float* hi;

    __device__ __forceinline__ float* at(int k, int c) const
    {
        return k == 0 ? dc + c : hi + (k - 1) * 3 + c;
    }

    __device__ __forceinline__ static ShGrad contiguous(float* p)
    {
        return {p, p + 3};
    }
};

// device_sh_eval_channel

/**
 * Raw, unbiased SH sum of channel c at an unnormalized direction.
 */
__device__ __forceinline__ float
device_sh_eval_channel(int degree, int c, float dx, float dy, float dz, ShCoeffs coeffs)
{
    // --- Degree 0 ---
    float result = SH_C0 * coeffs.at(0, c);
    if (degree < 1)
        return result;

    // Normalize view direction
    const float inorm = rsqrtf(dx * dx + dy * dy + dz * dz);
    const float x = dx * inorm;
    const float y = dy * inorm;
    const float z = dz * inorm;

    // --- Degree 1 --- (3 real basis functions
    result += SH_C1 * (-y * coeffs.at(1, c) + z * coeffs.at(2, c) - x * coeffs.at(3, c));
    if (degree < 2)
        return result;

    // --- Degree 2 --- (5 real basis functions)
    const float z2 = z * z;
    const float xy = x * y;
    const float xz = x * z;            // used in Y_2^1
    const float yz = y * z;            // used in Y_2^{-1}
    const float x2_y2 = x * x - y * y; // used in Y_2^2

    // P2_0 = z^2 − 1/3  →  Y_2^0  ∝  3z^2 − 1
    const float P2_0 = SH_C2_2 * z2 - SH_C2_2B;
    // Y_2^{-2} ∝ xy,  Y_2^{-1} ∝ yz,  Y_2^1 ∝ xz,  Y_2^2 ∝ (x^2−y^2)
    const float Y2_neg2 = SH_C2_0 * (2.f * xy);
    const float Y2_neg1 = -SH_C2_1 * yz;
    const float Y2_0 = P2_0;
    const float Y2_pos1 = -SH_C2_1 * xz;
    const float Y2_pos2 = SH_C2_0 * x2_y2;

    result += Y2_neg2 * coeffs.at(4, c) + Y2_neg1 * coeffs.at(5, c) + Y2_0 * coeffs.at(6, c) +
              Y2_pos1 * coeffs.at(7, c) + Y2_pos2 * coeffs.at(8, c);
    if (degree < 3)
        return result;

    // --- Degree 3 --- (7 real basis functions)
    const float z3 = z2 * z;
    // Auxiliary harmonic combinations
    const float fC2 = x * x2_y2 - y * (2.f * xy); // x(x²-y²) - y(2xy) = x³ - 3xy²
    const float fS2 = x * (2.f * xy) + y * x2_y2; // 2x²y + x²y - y³  = 3x²y - y³

    // Y_3^0  ∝ z(5z^2 − 3)
    const float Y3_0 = SH_C3_3 * z3 - SH_C3_3B * z;
    // Y_3^{±1} ∝ (5z^2−1)·{x,y}
    const float coeff_1 = -SH_C3_2 * z2 + SH_C3_2B;
    const float Y3_neg1 = coeff_1 * y;
    const float Y3_pos1 = coeff_1 * x;
    // Y_3^{±2} ∝ z·{xy, x^2−y^2}
    const float Y3_neg2 = SH_C3_1 * z * (2.f * xy);
    const float Y3_pos2 = SH_C3_1 * z * x2_y2;
    // Y_3^{±3} ∝ {3x^2y−y^3, x^3−3xy^2}
    const float Y3_neg3 = -SH_C3_0 * fS2;
    const float Y3_pos3 = -SH_C3_0 * fC2;

    result += Y3_neg3 * coeffs.at(9, c) + Y3_neg2 * coeffs.at(10, c) + Y3_neg1 * coeffs.at(11, c) +
              Y3_0 * coeffs.at(12, c) + Y3_pos1 * coeffs.at(13, c) + Y3_pos2 * coeffs.at(14, c) +
              Y3_pos3 * coeffs.at(15, c);

    return result;
}

// device_eval_sh4d_rgb

/**
 * color[c] = max(0, SH(dir)[c] + Σ_f sh_temporal[2f-2][c] cos(π f dir_t) + sh_temporal[2f-1][c] sin(π f dir_t) + 0.5).
 * Pass n_temporal_freqs = 0 and a null sh_temporal_ptr to skip the temporal term.
 */
__device__ __forceinline__ void device_eval_sh4d_rgb(
    int sh_degree,
    int K,
    int n_temporal_freqs,
    float dir_x,
    float dir_y,
    float dir_z,
    float dir_t,
    ShCoeffs sh,
    const float* __restrict__ sh_temporal_ptr,
    float& out_r,
    float& out_g,
    float& out_b
)
{
    // Spatial SH (raw, unbiased)
    float r = device_sh_eval_channel(sh_degree, 0, dir_x, dir_y, dir_z, sh);
    float g = device_sh_eval_channel(sh_degree, 1, dir_x, dir_y, dir_z, sh);
    float b = device_sh_eval_channel(sh_degree, 2, dir_x, dir_y, dir_z, sh);

    // Temporal additive term: interleaved cos/sin pairs per frequency
    if (sh_temporal_ptr != nullptr && n_temporal_freqs > 0) {
        for (int f = 1; f <= n_temporal_freqs; ++f) {
            const float angle = M_PI_F * (float)f * dir_t;
            const float cos_val = cosf(angle);
            const float sin_val = sinf(angle);
            // Coefficient layout: cos-coeff at k=(f-1)*2, sin-coeff at k=(f-1)*2+1
            const int k_cos = (f - 1) * 2;
            const int k_sin = k_cos + 1;
            r += sh_temporal_ptr[k_cos * 3 + 0] * cos_val + sh_temporal_ptr[k_sin * 3 + 0] * sin_val;
            g += sh_temporal_ptr[k_cos * 3 + 1] * cos_val + sh_temporal_ptr[k_sin * 3 + 1] * sin_val;
            b += sh_temporal_ptr[k_cos * 3 + 2] * cos_val + sh_temporal_ptr[k_sin * 3 + 2] * sin_val;
        }
    }

    // Bias and clamp (standard gsplat convention)
    out_r = fmaxf(0.0f, r + 0.5f);
    out_g = fmaxf(0.0f, g + 0.5f);
    out_b = fmaxf(0.0f, b + 0.5f);
}

// device_sh_coeff_grad

/**
 * Scatter dL/dcoeff[k] = d_raw_c · Y_k(dir) for the three channels, atomically (threads of several
 * cameras share a Gaussian). The direction is already normalized.
 *
 * Both leaves scatter through this: SH evaluation is linear in the coefficients, so a basis that
 * multiplies Y_k by a temporal weight calls it once per block with the upstream scaled by that weight.
 */
__device__ __forceinline__ void device_sh_coeff_grad(
    int degree,
    float x,
    float y,
    float z,
    float d_raw_r,
    float d_raw_g,
    float d_raw_b,
    ShGrad d_coeffs
)
{
    atomicAdd(d_coeffs.at(0, 0), d_raw_r * SH_C0);
    atomicAdd(d_coeffs.at(0, 1), d_raw_g * SH_C0);
    atomicAdd(d_coeffs.at(0, 2), d_raw_b * SH_C0);
    if (degree < 1)
        return;

    const float B1 = -SH_C1 * y, B2 = SH_C1 * z, B3 = -SH_C1 * x;
    atomicAdd(d_coeffs.at(1, 0), d_raw_r * B1);
    atomicAdd(d_coeffs.at(1, 1), d_raw_g * B1);
    atomicAdd(d_coeffs.at(1, 2), d_raw_b * B1);
    atomicAdd(d_coeffs.at(2, 0), d_raw_r * B2);
    atomicAdd(d_coeffs.at(2, 1), d_raw_g * B2);
    atomicAdd(d_coeffs.at(2, 2), d_raw_b * B2);
    atomicAdd(d_coeffs.at(3, 0), d_raw_r * B3);
    atomicAdd(d_coeffs.at(3, 1), d_raw_g * B3);
    atomicAdd(d_coeffs.at(3, 2), d_raw_b * B3);
    if (degree < 2)
        return;

    const float z2 = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float x2_y2 = x * x - y * y;
    const float Y2_neg2 = SH_C2_0 * (2.f * xy);
    const float Y2_neg1 = -SH_C2_1 * yz;
    const float Y2_0 = SH_C2_2 * z2 - SH_C2_2B;
    const float Y2_pos1 = -SH_C2_1 * xz;
    const float Y2_pos2 = SH_C2_0 * x2_y2;

    atomicAdd(d_coeffs.at(4, 0), d_raw_r * Y2_neg2);
    atomicAdd(d_coeffs.at(4, 1), d_raw_g * Y2_neg2);
    atomicAdd(d_coeffs.at(4, 2), d_raw_b * Y2_neg2);
    atomicAdd(d_coeffs.at(5, 0), d_raw_r * Y2_neg1);
    atomicAdd(d_coeffs.at(5, 1), d_raw_g * Y2_neg1);
    atomicAdd(d_coeffs.at(5, 2), d_raw_b * Y2_neg1);
    atomicAdd(d_coeffs.at(6, 0), d_raw_r * Y2_0);
    atomicAdd(d_coeffs.at(6, 1), d_raw_g * Y2_0);
    atomicAdd(d_coeffs.at(6, 2), d_raw_b * Y2_0);
    atomicAdd(d_coeffs.at(7, 0), d_raw_r * Y2_pos1);
    atomicAdd(d_coeffs.at(7, 1), d_raw_g * Y2_pos1);
    atomicAdd(d_coeffs.at(7, 2), d_raw_b * Y2_pos1);
    atomicAdd(d_coeffs.at(8, 0), d_raw_r * Y2_pos2);
    atomicAdd(d_coeffs.at(8, 1), d_raw_g * Y2_pos2);
    atomicAdd(d_coeffs.at(8, 2), d_raw_b * Y2_pos2);
    if (degree < 3)
        return;

    const float z3 = z2 * z;
    const float fC2 = x * x2_y2 - y * (2.f * xy);
    const float fS2 = x * (2.f * xy) + y * x2_y2;
    const float coeff_1 = -SH_C3_2 * z2 + SH_C3_2B;
    const float Y3_neg3 = -SH_C3_0 * fS2;
    const float Y3_neg2 = SH_C3_1 * z * (2.f * xy);
    const float Y3_neg1 = coeff_1 * y;
    const float Y3_0 = SH_C3_3 * z3 - SH_C3_3B * z;
    const float Y3_pos1 = coeff_1 * x;
    const float Y3_pos2 = SH_C3_1 * z * x2_y2;
    const float Y3_pos3 = -SH_C3_0 * fC2;

    atomicAdd(d_coeffs.at(9, 0), d_raw_r * Y3_neg3);
    atomicAdd(d_coeffs.at(9, 1), d_raw_g * Y3_neg3);
    atomicAdd(d_coeffs.at(9, 2), d_raw_b * Y3_neg3);
    atomicAdd(d_coeffs.at(10, 0), d_raw_r * Y3_neg2);
    atomicAdd(d_coeffs.at(10, 1), d_raw_g * Y3_neg2);
    atomicAdd(d_coeffs.at(10, 2), d_raw_b * Y3_neg2);
    atomicAdd(d_coeffs.at(11, 0), d_raw_r * Y3_neg1);
    atomicAdd(d_coeffs.at(11, 1), d_raw_g * Y3_neg1);
    atomicAdd(d_coeffs.at(11, 2), d_raw_b * Y3_neg1);
    atomicAdd(d_coeffs.at(12, 0), d_raw_r * Y3_0);
    atomicAdd(d_coeffs.at(12, 1), d_raw_g * Y3_0);
    atomicAdd(d_coeffs.at(12, 2), d_raw_b * Y3_0);
    atomicAdd(d_coeffs.at(13, 0), d_raw_r * Y3_pos1);
    atomicAdd(d_coeffs.at(13, 1), d_raw_g * Y3_pos1);
    atomicAdd(d_coeffs.at(13, 2), d_raw_b * Y3_pos1);
    atomicAdd(d_coeffs.at(14, 0), d_raw_r * Y3_pos2);
    atomicAdd(d_coeffs.at(14, 1), d_raw_g * Y3_pos2);
    atomicAdd(d_coeffs.at(14, 2), d_raw_b * Y3_pos2);
    atomicAdd(d_coeffs.at(15, 0), d_raw_r * Y3_pos3);
    atomicAdd(d_coeffs.at(15, 1), d_raw_g * Y3_pos3);
    atomicAdd(d_coeffs.at(15, 2), d_raw_b * Y3_pos3);
}

// device_sh_dir_grad

/**
 * Accumulate the SH contribution to dL/d(normalized direction) for one coefficient block.
 * Each line is the derivative of the matching Y_k in device_sh_eval_channel; the caller folds in
 * the normalization Jacobian once, through device_dir_norm_bwd.
 */
__device__ __forceinline__ void device_sh_dir_grad(
    int degree,
    float x,
    float y,
    float z,
    ShCoeffs coeffs,
    float d_raw_r,
    float d_raw_g,
    float d_raw_b,
    float& d_dir_xn,
    float& d_dir_yn,
    float& d_dir_zn
)
{
    // Degree 0 is constant in the direction.
    if (degree < 1)
        return;

    const float c1_dot = coeffs.at(1, 0) * d_raw_r + coeffs.at(1, 1) * d_raw_g + coeffs.at(1, 2) * d_raw_b;
    const float c2_dot = coeffs.at(2, 0) * d_raw_r + coeffs.at(2, 1) * d_raw_g + coeffs.at(2, 2) * d_raw_b;
    const float c3_dot = coeffs.at(3, 0) * d_raw_r + coeffs.at(3, 1) * d_raw_g + coeffs.at(3, 2) * d_raw_b;
    d_dir_xn += -SH_C1 * c3_dot;
    d_dir_yn += -SH_C1 * c1_dot;
    d_dir_zn += SH_C1 * c2_dot;
    if (degree < 2)
        return;

    const float z2 = z * z;
    const float x2 = x * x, y2 = y * y;
    const float x2_y2 = x2 - y2;

    const float c4_dot = coeffs.at(4, 0) * d_raw_r + coeffs.at(4, 1) * d_raw_g + coeffs.at(4, 2) * d_raw_b;
    const float c5_dot = coeffs.at(5, 0) * d_raw_r + coeffs.at(5, 1) * d_raw_g + coeffs.at(5, 2) * d_raw_b;
    const float c6_dot = coeffs.at(6, 0) * d_raw_r + coeffs.at(6, 1) * d_raw_g + coeffs.at(6, 2) * d_raw_b;
    const float c7_dot = coeffs.at(7, 0) * d_raw_r + coeffs.at(7, 1) * d_raw_g + coeffs.at(7, 2) * d_raw_b;
    const float c8_dot = coeffs.at(8, 0) * d_raw_r + coeffs.at(8, 1) * d_raw_g + coeffs.at(8, 2) * d_raw_b;
    // Y2_neg2=SH_C2_0*2xy: d_x+=2SH_C2_0*y, d_y+=2SH_C2_0*x
    // Y2_neg1=-SH_C2_1*yz: d_y+=-SH_C2_1*z, d_z+=-SH_C2_1*y
    // Y2_0=SH_C2_2*z^2-SH_C2_2B: d_z+=2*SH_C2_2*z
    // Y2_pos1=-SH_C2_1*xz: d_x+=-SH_C2_1*z, d_z+=-SH_C2_1*x
    // Y2_pos2=SH_C2_0*(x^2-y^2): d_x+=2*SH_C2_0*x, d_y+=-2*SH_C2_0*y
    d_dir_xn += SH_C2_0 * 2.0f * y * c4_dot - SH_C2_1 * z * c7_dot + SH_C2_0 * 2.0f * x * c8_dot;
    d_dir_yn += SH_C2_0 * 2.0f * x * c4_dot - SH_C2_1 * z * c5_dot - SH_C2_0 * 2.0f * y * c8_dot;
    d_dir_zn += -SH_C2_1 * y * c5_dot + 2.0f * SH_C2_2 * z * c6_dot - SH_C2_1 * x * c7_dot;
    if (degree < 3)
        return;

    const float coeff_1 = -SH_C3_2 * z2 + SH_C3_2B;

    const float c9_dot = coeffs.at(9, 0) * d_raw_r + coeffs.at(9, 1) * d_raw_g + coeffs.at(9, 2) * d_raw_b;
    const float c10_dot = coeffs.at(10, 0) * d_raw_r + coeffs.at(10, 1) * d_raw_g + coeffs.at(10, 2) * d_raw_b;
    const float c11_dot = coeffs.at(11, 0) * d_raw_r + coeffs.at(11, 1) * d_raw_g + coeffs.at(11, 2) * d_raw_b;
    const float c12_dot = coeffs.at(12, 0) * d_raw_r + coeffs.at(12, 1) * d_raw_g + coeffs.at(12, 2) * d_raw_b;
    const float c13_dot = coeffs.at(13, 0) * d_raw_r + coeffs.at(13, 1) * d_raw_g + coeffs.at(13, 2) * d_raw_b;
    const float c14_dot = coeffs.at(14, 0) * d_raw_r + coeffs.at(14, 1) * d_raw_g + coeffs.at(14, 2) * d_raw_b;
    const float c15_dot = coeffs.at(15, 0) * d_raw_r + coeffs.at(15, 1) * d_raw_g + coeffs.at(15, 2) * d_raw_b;
    // Y3_neg3=-SH_C3_0*fS2, fS2=3x^2y-y^3: d_x+=-SH_C3_0*6xy, d_y+=-SH_C3_0*(3x^2-3y^2)
    d_dir_xn += -SH_C3_0 * 6.0f * x * y * c9_dot;
    d_dir_yn += -SH_C3_0 * (3.0f * x2 - 3.0f * y2) * c9_dot;
    // Y3_neg2=SH_C3_1*2xyz: d_x+=SH_C3_1*2yz, d_y+=SH_C3_1*2xz, d_z+=SH_C3_1*2xy
    d_dir_xn += SH_C3_1 * 2.0f * y * z * c10_dot;
    d_dir_yn += SH_C3_1 * 2.0f * x * z * c10_dot;
    d_dir_zn += SH_C3_1 * 2.0f * x * y * c10_dot;
    // Y3_neg1=coeff_1*y: d_y+=coeff_1, d_z+=-2*SH_C3_2*z*y
    d_dir_yn += coeff_1 * c11_dot;
    d_dir_zn += -2.0f * SH_C3_2 * z * y * c11_dot;
    // Y3_0=SH_C3_3*z^3-SH_C3_3B*z: d_z+=3*SH_C3_3*z^2-SH_C3_3B
    d_dir_zn += (3.0f * SH_C3_3 * z2 - SH_C3_3B) * c12_dot;
    // Y3_pos1=coeff_1*x: d_x+=coeff_1, d_z+=-2*SH_C3_2*z*x
    d_dir_xn += coeff_1 * c13_dot;
    d_dir_zn += -2.0f * SH_C3_2 * z * x * c13_dot;
    // Y3_pos2=SH_C3_1*z*(x^2-y^2): d_x+=SH_C3_1*2xz, d_y+=SH_C3_1*(-2yz), d_z+=SH_C3_1*(x^2-y^2)
    d_dir_xn += SH_C3_1 * 2.0f * x * z * c14_dot;
    d_dir_yn += SH_C3_1 * (-2.0f) * y * z * c14_dot;
    d_dir_zn += SH_C3_1 * x2_y2 * c14_dot;
    // Y3_pos3=-SH_C3_0*fC2, fC2=x^3-3xy^2: d_x+=-SH_C3_0*(3x^2-3y^2), d_y+=-SH_C3_0*(-6xy)
    d_dir_xn += -SH_C3_0 * (3.0f * x2 - 3.0f * y2) * c15_dot;
    d_dir_yn += -SH_C3_0 * (-6.0f * x * y) * c15_dot;
}

// device_dir_norm_bwd

/**
 * Fold the direction-normalization Jacobian: d_dir_unnorm += (I - n nᵀ) · inorm · d_dir_norm.
 */
__device__ __forceinline__ void device_dir_norm_bwd(
    float x,
    float y,
    float z,
    float inorm,
    float d_dir_xn,
    float d_dir_yn,
    float d_dir_zn,
    float& d_dir_x,
    float& d_dir_y,
    float& d_dir_z
)
{
    const float x_dot_d = x * d_dir_xn + y * d_dir_yn + z * d_dir_zn;
    d_dir_x += (d_dir_xn - x * x_dot_d) * inorm;
    d_dir_y += (d_dir_yn - y * x_dot_d) * inorm;
    d_dir_z += (d_dir_zn - z * x_dot_d) * inorm;
}

// device_eval_sh4d_rgb_bwd

/**
 * Backward of device_eval_sh4d_rgb through the ReLU gate and direction normalization.
 * SH gradients are atomicAdd-ed (several cameras' threads share a Gaussian); direction gradients use +=.
 */
__device__ __forceinline__ void device_eval_sh4d_rgb_bwd(
    int sh_degree,
    int K,
    int n_temporal_freqs,
    float dir_x,
    float dir_y,
    float dir_z,
    float dir_t,
    ShCoeffs sh,
    const float* __restrict__ sh_temporal_ptr,
    float up_dr,
    float up_dg,
    float up_db,
    ShGrad d_sh,
    float* d_sh_temporal_ptr,
    float& d_dir_x,
    float& d_dir_y,
    float& d_dir_z
)
{
    // --- Recompute the raw colour (needed for the ReLU gate) ---
    const float raw_r = device_sh_eval_channel(sh_degree, 0, dir_x, dir_y, dir_z, sh);
    const float raw_g = device_sh_eval_channel(sh_degree, 1, dir_x, dir_y, dir_z, sh);
    const float raw_b = device_sh_eval_channel(sh_degree, 2, dir_x, dir_y, dir_z, sh);

    float temporal_r = 0.0f, temporal_g = 0.0f, temporal_b = 0.0f;
    if (sh_temporal_ptr != nullptr && n_temporal_freqs > 0) {
        for (int f = 1; f <= n_temporal_freqs; ++f) {
            const float angle = M_PI_F * (float)f * dir_t;
            const int k_cos = (f - 1) * 2;
            const int k_sin = k_cos + 1;
            temporal_r += sh_temporal_ptr[k_cos * 3 + 0] * cosf(angle) + sh_temporal_ptr[k_sin * 3 + 0] * sinf(angle);
            temporal_g += sh_temporal_ptr[k_cos * 3 + 1] * cosf(angle) + sh_temporal_ptr[k_sin * 3 + 1] * sinf(angle);
            temporal_b += sh_temporal_ptr[k_cos * 3 + 2] * cosf(angle) + sh_temporal_ptr[k_sin * 3 + 2] * sinf(angle);
        }
    }

    // ReLU gate: color = max(0, raw + temporal + 0.5)
    const float d_raw_r = (raw_r + temporal_r + 0.5f > 0.0f) ? up_dr : 0.0f;
    const float d_raw_g = (raw_g + temporal_g + 0.5f > 0.0f) ? up_dg : 0.0f;
    const float d_raw_b = (raw_b + temporal_b + 0.5f > 0.0f) ? up_db : 0.0f;

    // --- Backward through the temporal Fourier term ---
    if (sh_temporal_ptr != nullptr && n_temporal_freqs > 0 && d_sh_temporal_ptr != nullptr) {
        for (int f = 1; f <= n_temporal_freqs; ++f) {
            const float angle = M_PI_F * (float)f * dir_t;
            const float cos_val = cosf(angle);
            const float sin_val = sinf(angle);
            const int k_cos = (f - 1) * 2;
            const int k_sin = k_cos + 1;
            atomicAdd(&d_sh_temporal_ptr[k_cos * 3 + 0], d_raw_r * cos_val);
            atomicAdd(&d_sh_temporal_ptr[k_cos * 3 + 1], d_raw_g * cos_val);
            atomicAdd(&d_sh_temporal_ptr[k_cos * 3 + 2], d_raw_b * cos_val);
            atomicAdd(&d_sh_temporal_ptr[k_sin * 3 + 0], d_raw_r * sin_val);
            atomicAdd(&d_sh_temporal_ptr[k_sin * 3 + 1], d_raw_g * sin_val);
            atomicAdd(&d_sh_temporal_ptr[k_sin * 3 + 2], d_raw_b * sin_val);
        }
    }

    // --- Backward through the spatial SH, on the normalized direction ---
    const float inorm = rsqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    const float x = dir_x * inorm;
    const float y = dir_y * inorm;
    const float z = dir_z * inorm;

    device_sh_coeff_grad(sh_degree, x, y, z, d_raw_r, d_raw_g, d_raw_b, d_sh);

    float d_dir_xn = 0.0f, d_dir_yn = 0.0f, d_dir_zn = 0.0f;
    device_sh_dir_grad(sh_degree, x, y, z, sh, d_raw_r, d_raw_g, d_raw_b, d_dir_xn, d_dir_yn, d_dir_zn);
    device_dir_norm_bwd(x, y, z, inorm, d_dir_xn, d_dir_yn, d_dir_zn, d_dir_x, d_dir_y, d_dir_z);
}

// AppearanceOps<M>

/**
 * The colour leaf, selected at compile time — the appearance counterpart of CameraOps<F>.
 *
 * Each specialization owns one temporal basis. ``n_terms`` counts that basis's active terms
 * (additive: frequencies; multiplicative: temporal orders), ``coeff_stride`` is the floats per
 * Gaussian the host laid out, and ``coeffs`` points at this Gaussian's temporal coefficients,
 * already offset by that stride; it may be null, which means "spatial SH only".
 *
 * Neither leaf differentiates ``dir_t``: colour sends no gradient to mu_t.
 */
template <AppearanceKind M> struct AppearanceOps; // primary template — undefined

template <> struct AppearanceOps<AppearanceKind::Additive> {
    __device__ __forceinline__ static void eval(
        int sh_degree,
        int n_terms,
        int coeff_stride,
        float dir_x,
        float dir_y,
        float dir_z,
        float dir_t,
        ShCoeffs sh,
        const float* __restrict__ coeffs,
        float& out_r,
        float& out_g,
        float& out_b
    )
    {
        (void)coeff_stride; // the interleaved cos/sin layout follows from n_terms alone
        const int K = (sh_degree + 1) * (sh_degree + 1);
        device_eval_sh4d_rgb(sh_degree, K, n_terms, dir_x, dir_y, dir_z, dir_t, sh, coeffs, out_r, out_g, out_b);
    }

    __device__ __forceinline__ static void eval_bwd(
        int sh_degree,
        int n_terms,
        int coeff_stride,
        float dir_x,
        float dir_y,
        float dir_z,
        float dir_t,
        ShCoeffs sh,
        const float* __restrict__ coeffs,
        float up_dr,
        float up_dg,
        float up_db,
        ShGrad d_sh,
        float* d_coeffs,
        float& d_dir_x,
        float& d_dir_y,
        float& d_dir_z
    )
    {
        (void)coeff_stride;
        const int K = (sh_degree + 1) * (sh_degree + 1);
        device_eval_sh4d_rgb_bwd(
            sh_degree,
            K,
            n_terms,
            dir_x,
            dir_y,
            dir_z,
            dir_t,
            sh,
            coeffs,
            up_dr,
            up_dg,
            up_db,
            d_sh,
            d_coeffs,
            d_dir_x,
            d_dir_y,
            d_dir_z
        );
    }
};

/**
 * Published 4DGS 4D SH (utils/sh_utils.py:eval_shfs_4d): every spatial harmonic carries its own
 * temporal modulation, so view dependence itself varies with time.
 *
 *   color = max(0, Σ_{n=0..n_terms} cos(2π n dir_t) · Σ_k Y_k(dir) · coeff[n, k] + 0.5)
 *
 * cos only, no sine term. Block n-1 of ``coeffs`` holds the spatial coefficients of temporal order
 * n, DC first, at the maximum K, exactly as the published tensor is indexed.
 *
 * The sum is linear in the coefficients, so each order is one more call of the shared spatial
 * evaluator, weighted; no basis table and no per-thread scratch.
 */
template <> struct AppearanceOps<AppearanceKind::Multiplicative> {
    __device__ __forceinline__ static void eval(
        int sh_degree,
        int n_terms,
        int coeff_stride,
        float dir_x,
        float dir_y,
        float dir_z,
        float dir_t,
        ShCoeffs sh,
        const float* __restrict__ coeffs,
        float& out_r,
        float& out_g,
        float& out_b
    )
    {
        float r = device_sh_eval_channel(sh_degree, 0, dir_x, dir_y, dir_z, sh);
        float g = device_sh_eval_channel(sh_degree, 1, dir_x, dir_y, dir_z, sh);
        float b = device_sh_eval_channel(sh_degree, 2, dir_x, dir_y, dir_z, sh);

        if (coeffs != nullptr && n_terms > 0) {
            const int block = coeff_stride / MAX_TEMPORAL_DEGREE;
#pragma unroll
            for (int n = 1; n <= MAX_TEMPORAL_DEGREE; ++n) {
                if (n > n_terms)
                    break;
                const float w = cosf(2.f * M_PI_F * (float)n * dir_t);
                const ShCoeffs c = ShCoeffs::contiguous(coeffs + (n - 1) * block);
                r += w * device_sh_eval_channel(sh_degree, 0, dir_x, dir_y, dir_z, c);
                g += w * device_sh_eval_channel(sh_degree, 1, dir_x, dir_y, dir_z, c);
                b += w * device_sh_eval_channel(sh_degree, 2, dir_x, dir_y, dir_z, c);
            }
        }

        out_r = fmaxf(0.0f, r + 0.5f);
        out_g = fmaxf(0.0f, g + 0.5f);
        out_b = fmaxf(0.0f, b + 0.5f);
    }

    __device__ __forceinline__ static void eval_bwd(
        int sh_degree,
        int n_terms,
        int coeff_stride,
        float dir_x,
        float dir_y,
        float dir_z,
        float dir_t,
        ShCoeffs sh,
        const float* __restrict__ coeffs,
        float up_dr,
        float up_dg,
        float up_db,
        ShGrad d_sh,
        float* d_coeffs,
        float& d_dir_x,
        float& d_dir_y,
        float& d_dir_z
    )
    {
        // Raw colour, for the ReLU gate.
        float raw_r, raw_g, raw_b;
        eval(sh_degree, n_terms, coeff_stride, dir_x, dir_y, dir_z, dir_t, sh, coeffs, raw_r, raw_g, raw_b);
        const float d_raw_r = (raw_r > 0.0f) ? up_dr : 0.0f;
        const float d_raw_g = (raw_g > 0.0f) ? up_dg : 0.0f;
        const float d_raw_b = (raw_b > 0.0f) ? up_db : 0.0f;

        const float inorm = rsqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
        const float x = dir_x * inorm;
        const float y = dir_y * inorm;
        const float z = dir_z * inorm;

        float d_dir_xn = 0.0f, d_dir_yn = 0.0f, d_dir_zn = 0.0f;

        // Temporal order 0 is the spatial block, at weight 1.
        device_sh_coeff_grad(sh_degree, x, y, z, d_raw_r, d_raw_g, d_raw_b, d_sh);
        device_sh_dir_grad(sh_degree, x, y, z, sh, d_raw_r, d_raw_g, d_raw_b, d_dir_xn, d_dir_yn, d_dir_zn);

        // d_coeffs gates only the scatter: the temporal orders still steer the direction gradient.
        if (coeffs != nullptr && n_terms > 0) {
            const int block = coeff_stride / MAX_TEMPORAL_DEGREE;
#pragma unroll
            for (int n = 1; n <= MAX_TEMPORAL_DEGREE; ++n) {
                if (n > n_terms)
                    break;
                const float w = cosf(2.f * M_PI_F * (float)n * dir_t);
                const ShCoeffs c = ShCoeffs::contiguous(coeffs + (n - 1) * block);
                if (d_coeffs != nullptr) {
                    device_sh_coeff_grad(
                        sh_degree,
                        x,
                        y,
                        z,
                        w * d_raw_r,
                        w * d_raw_g,
                        w * d_raw_b,
                        ShGrad::contiguous(d_coeffs + (n - 1) * block)
                    );
                }
                device_sh_dir_grad(
                    sh_degree,
                    x,
                    y,
                    z,
                    c,
                    w * d_raw_r,
                    w * d_raw_g,
                    w * d_raw_b,
                    d_dir_xn,
                    d_dir_yn,
                    d_dir_zn
                );
            }
        }

        device_dir_norm_bwd(x, y, z, inorm, d_dir_xn, d_dir_yn, d_dir_zn, d_dir_x, d_dir_y, d_dir_z);
    }
};
