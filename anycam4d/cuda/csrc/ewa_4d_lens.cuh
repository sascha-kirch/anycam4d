/**
 * EWA through a distorted lens: the mean through CameraOps<F>::project, the covariance through the lens's
 * local Jacobian J = diag(f) · J_D(p̄) · J_P. Both families are n = ψ(u)·p + t(p) on the pinhole-normalised
 * p = (x/z, y/z), u = |p|²; EwaLens<F> supplies ψ, ψ′, ψ″ and t.
 * The undistorted PINHOLE camera never comes here; it takes device_persp_proj_ewa.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "camera_ops.cuh"
#include "common.h"
#include "ewa_4d.cuh"

template <ProjectionFamily F> struct EwaLens; // primary template — undefined

/** One camera's lens: its distortion row and, for Brown–Conrady, its clamp limits [x_pos, x_neg, y_pos, y_neg]. */
struct EwaLensCam {
    const float* dist;
    const float* lims;
};

// Brown–Conrady: ψ = N/D rational radial, tangential p1, p2 and thin prism s1..s4.
template <> struct EwaLens<PINHOLE_FAMILY> {
    // The pinhole frustum clamp on p (gsplat's persp_proj), so off-screen centres keep a bounded J.
    static constexpr bool kClamp = true;

    /** ψ, ψ′, ψ″ at u; false outside the monotone radial domain (CameraOps::project's test). */
    static __device__ __forceinline__ bool radial(float u, const float* d, float& p0, float& p1, float& p2)
    {
        const float n = 1.f + u * (d[0] + u * (d[1] + u * d[4]));
        const float m = 1.f + u * (d[5] + u * (d[6] + u * d[7]));
        const float n1 = d[0] + u * (2.f * d[1] + u * 3.f * d[4]);
        const float m1 = d[5] + u * (2.f * d[6] + u * 3.f * d[7]);
        if (m <= 1e-8f || (n * m + 2.f * u * (n1 * m - n * m1)) <= 0.f)
            return false;
        const float n2 = 2.f * d[1] + 6.f * u * d[4];
        const float m2 = 2.f * d[6] + 6.f * u * d[7];
        const float im = 1.f / m;
        p0 = n * im;
        p1 = (n1 * m - n * m1) * im * im;
        p2 = (n2 * m - n * m2) * im * im - 2.f * m1 * p1 * im;
        return true;
    }

    /** J_D += ∂t/∂p (row-major 2×2). */
    static __device__ __forceinline__ void add_tangential_jacobian(float x, float y, const float* d, float jd[4])
    {
        const float u = x * x + y * y;
        const float sx = d[8] + 2.f * d[9] * u, sy = d[10] + 2.f * d[11] * u;
        jd[0] += 2.f * d[2] * y + 6.f * d[3] * x + 2.f * x * sx;
        jd[1] += 2.f * d[2] * x + 2.f * d[3] * y + 2.f * y * sx;
        jd[2] += 2.f * d[2] * x + 2.f * d[3] * y + 2.f * x * sy;
        jd[3] += 6.f * d[2] * y + 2.f * d[3] * x + 2.f * y * sy;
    }

    /** g_k += Σ_ij A_ij ∂²t_i/∂p_j∂p_k. */
    static __device__ __forceinline__ void
    add_tangential_contraction(float x, float y, const float* d, const float a[4], float& gx, float& gy)
    {
        const float u = x * x + y * y;
        const float hx = 2.f * d[8] + 4.f * d[9] * u, hy = 2.f * d[10] + 4.f * d[11] * u;
        // t_x: ∂xx, ∂xy, ∂yy
        const float tx_xx = 6.f * d[3] + hx + 8.f * d[9] * x * x;
        const float tx_xy = 2.f * d[2] + 8.f * d[9] * x * y;
        const float tx_yy = 2.f * d[3] + hx + 8.f * d[9] * y * y;
        // t_y: ∂xx, ∂xy, ∂yy
        const float ty_xx = 2.f * d[2] + hy + 8.f * d[11] * x * x;
        const float ty_xy = 2.f * d[3] + 8.f * d[11] * x * y;
        const float ty_yy = 6.f * d[2] + hy + 8.f * d[11] * y * y;
        gx += a[0] * tx_xx + a[1] * tx_xy + a[2] * ty_xx + a[3] * ty_xy;
        gy += a[0] * tx_xy + a[1] * tx_yy + a[2] * ty_xy + a[3] * ty_yy;
    }
};

// Kannala–Brandt: ψ(u) = θd(θ)/tan θ with θ = atan √u; purely radial.
template <> struct EwaLens<FISHEYE_FAMILY> {
    static constexpr bool kClamp = false; // as gsplat's fisheye_proj

    // Below this u (θ < 5.7°) the closed-form ψ′, ψ″ cancel; the series through u⁴ is used instead.
    static constexpr float U_SERIES = 1e-2f;

    /** ψ, ψ′, ψ″ at u; false past the fold dθd/dθ ≤ 0 (CameraOps::radial's test). */
    static __device__ __forceinline__ bool radial(float u, const float* d, float& p0, float& p1, float& p2)
    {
        const float k1 = d[0], k2 = d[1], k3 = d[2], k4 = d[3];
        const float su = sqrtf(u);
        const float th = atanf(su);
        const float t2 = th * th;
        const float d1 = 1.f + t2 * (3.f * k1 + t2 * (5.f * k2 + t2 * (7.f * k3 + t2 * 9.f * k4)));
        if (d1 <= 0.f)
            return false;
        if (u < U_SERIES) {
            const float c1 = k1 - 1.f / 3.f;
            const float c2 = k2 - k1 + 1.f / 5.f;
            const float c3 = k3 - 5.f / 3.f * k2 + 14.f / 15.f * k1 - 1.f / 7.f;
            const float c4 = k4 - 7.f / 3.f * k3 + 19.f / 9.f * k2 - 818.f / 945.f * k1 + 1.f / 9.f;
            p0 = 1.f + u * (c1 + u * (c2 + u * (c3 + u * c4)));
            p1 = c1 + u * (2.f * c2 + u * (3.f * c3 + u * 4.f * c4));
            p2 = 2.f * c2 + u * (6.f * c3 + u * 12.f * c4);
            return true;
        }
        const float d0 = th * (1.f + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
        const float d2 = th * (6.f * k1 + t2 * (20.f * k2 + t2 * (42.f * k3 + t2 * 72.f * k4)));
        const float iu = 1.f / u, isu = 1.f / su, i1u = 1.f / (1.f + u);
        const float th_u = 0.5f * isu * i1u;
        p0 = d0 * isu;
        p1 = 0.5f * iu * (d1 * i1u - p0);
        p2 = 0.5f * iu * (d2 * th_u * i1u - d1 * i1u * i1u - d1 * th_u * isu + 0.5f * p0 * iu) - p1 * iu;
        return true;
    }

    static __device__ __forceinline__ void add_tangential_jacobian(float, float, const float*, float*) {}

    static __device__ __forceinline__ void
    add_tangential_contraction(float, float, const float*, const float*, float&, float&)
    {
    }
};

/** J_D(p) = ψ I + 2ψ′ p pᵀ + ∂t/∂p (row-major 2×2), and ψ′, ψ″; false outside the lens domain. */
template <ProjectionFamily F>
__device__ __forceinline__ bool device_lens_jd(float x, float y, const float* d, float jd[4], float& p1, float& p2)
{
    float p0;
    if (!EwaLens<F>::radial(x * x + y * y, d, p0, p1, p2))
        return false;
    jd[0] = p0 + 2.f * p1 * x * x;
    jd[1] = 2.f * p1 * x * y;
    jd[2] = jd[1];
    jd[3] = p0 + 2.f * p1 * y * y;
    EwaLens<F>::add_tangential_jacobian(x, y, d, jd);
    return true;
}

/** Frustum clamp of p at the camera's undistorted limits; flags mark a clamped axis. */
__device__ __forceinline__ void
device_ewa_frustum_clamp(const float* lims, float& px, float& py, bool& clamped_x, bool& clamped_y)
{
    clamped_x = px > lims[0] || px < -lims[1];
    clamped_y = py > lims[2] || py < -lims[3];
    px = fminf(lims[0], fmaxf(-lims[1], px));
    py = fminf(lims[2], fmaxf(-lims[3], py));
}

/**
 * Covariance Jacobian J (row-major 2×3) at p̄ and the pieces its backward reuses.
 * Returns false when the centre is outside the lens domain.
 */
template <ProjectionFamily F> struct LensJacobian {
    float j[6];
    float jd[4];
    float px, py; // p̄, the point J_D and J_P are evaluated at
    float psi1, psi2;
    bool clamped_x, clamped_y;

    __device__ __forceinline__ bool compute(float mc_x, float mc_y, float rz, float fx, float fy, EwaLensCam lens)
    {
        px = mc_x * rz;
        py = mc_y * rz;
        clamped_x = clamped_y = false;
        if constexpr (EwaLens<F>::kClamp)
            device_ewa_frustum_clamp(lens.lims, px, py, clamped_x, clamped_y);
        if (!device_lens_jd<F>(px, py, lens.dist, jd, psi1, psi2))
            return false;
        // J = diag(f) · J_D · [[1/z, 0, −p̄x/z], [0, 1/z, −p̄y/z]]
        j[0] = fx * rz * jd[0];
        j[1] = fx * rz * jd[1];
        j[2] = -(j[0] * px + j[1] * py);
        j[3] = fy * rz * jd[2];
        j[4] = fy * rz * jd[3];
        j[5] = -(j[3] * px + j[4] * py);
        return true;
    }
};

/** Σ2d = J Σ_c Jᵀ for row-major J [2×3] and upper-tri Σ_c. */
__device__ __forceinline__ void device_lens_covar2d(
    const float j[6],
    float cc00,
    float cc01,
    float cc02,
    float cc11,
    float cc12,
    float cc22,
    float& c00,
    float& c01,
    float& c11
)
{
    const float a0 = j[0] * cc00 + j[1] * cc01 + j[2] * cc02;
    const float a1 = j[0] * cc01 + j[1] * cc11 + j[2] * cc12;
    const float a2 = j[0] * cc02 + j[1] * cc12 + j[2] * cc22;
    const float b0 = j[3] * cc00 + j[4] * cc01 + j[5] * cc02;
    const float b1 = j[3] * cc01 + j[4] * cc11 + j[5] * cc12;
    const float b2 = j[3] * cc02 + j[4] * cc12 + j[5] * cc22;
    c00 = a0 * j[0] + a1 * j[1] + a2 * j[2];
    c01 = a0 * j[3] + a1 * j[4] + a2 * j[5];
    c11 = b0 * j[3] + b1 * j[4] + b2 * j[5];
}

/** device_persp_proj_ewa through the lens of family F (dist: that camera's n_slots coefficients). */
template <ProjectionFamily F>
__device__ __forceinline__ bool device_lens_proj_ewa(
    float mc_x,
    float mc_y,
    float mc_z,
    float cc00,
    float cc01,
    float cc02,
    float cc11,
    float cc12,
    float cc22,
    float fx,
    float fy,
    float cx,
    float cy,
    EwaLensCam lens,
    float eps2d,
    float opacity_cond,
    float near_plane,
    float far_plane,
    int image_width,
    int image_height,
    bool calc_compensations,
    float& out_mean2d_x,
    float& out_mean2d_y,
    float& out_conic_a,
    float& out_conic_b,
    float& out_conic_c,
    float& out_depth,
    int32_t& out_radius_x,
    int32_t& out_radius_y,
    float& out_compensation
)
{
    if (mc_z < near_plane || mc_z > far_plane)
        return false;
    out_depth = mc_z;
    if (!CameraOps<F>::project(mc_x, mc_y, mc_z, fx, fy, cx, cy, lens.dist, out_mean2d_x, out_mean2d_y))
        return false;

    LensJacobian<F> lj;
    if (!lj.compute(mc_x, mc_y, 1.0f / mc_z, fx, fy, lens))
        return false;

    float c2d00, c2d01, c2d11;
    device_lens_covar2d(lj.j, cc00, cc01, cc02, cc11, cc12, cc22, c2d00, c2d01, c2d11);

    const float det_orig = calc_compensations ? (c2d00 * c2d11 - c2d01 * c2d01) : 0.0f;
    c2d00 += eps2d;
    c2d11 += eps2d;
    const float det = c2d00 * c2d11 - c2d01 * c2d01;
    if (det <= 0.0f)
        return false;
    out_compensation = calc_compensations ? sqrtf(fmaxf(0.0f, det_orig / det)) : 1.0f;

    if (opacity_cond < ALPHA_THRESHOLD)
        return false;
    const float extend = fminf(3.33f, sqrtf(2.0f * __logf(opacity_cond / ALPHA_THRESHOLD)));
    out_radius_x = (int32_t)(ceilf(extend * sqrtf(c2d00)));
    out_radius_y = (int32_t)(ceilf(extend * sqrtf(c2d11)));

    const float w = (float)image_width;
    const float h = (float)image_height;
    if (out_mean2d_x + (float)out_radius_x <= 0.0f || out_mean2d_x - (float)out_radius_x >= w ||
        out_mean2d_y + (float)out_radius_y <= 0.0f || out_mean2d_y - (float)out_radius_y >= h)
        return false;

    const float inv_det = 1.0f / det;
    out_conic_a = c2d11 * inv_det;
    out_conic_b = -c2d01 * inv_det;
    out_conic_c = c2d00 * inv_det;
    return true;
}

/**
 * Backward of device_lens_proj_ewa: recompute and cull identically, then += gradients to the camera-space
 * mean (projection + Hessian contraction through J) and covariance. Same contract as device_persp_proj_ewa_bwd.
 */
template <ProjectionFamily F>
__device__ __forceinline__ bool device_lens_proj_ewa_bwd(
    float mc_x,
    float mc_y,
    float mc_z,
    float cc00,
    float cc01,
    float cc02,
    float cc11,
    float cc12,
    float cc22,
    float fx,
    float fy,
    float cx,
    float cy,
    EwaLensCam lens,
    float eps2d,
    float opacity_cond,
    float near_plane,
    float far_plane,
    int image_width,
    int image_height,
    float up_dm2d_x,
    float up_dm2d_y,
    float up_da,
    float up_db,
    float up_dc,
    bool calc_compensations,
    float up_dopac,
    float& compensation,
    float& d_mc_x,
    float& d_mc_y,
    float& d_mc_z,
    float& d_cc00,
    float& d_cc01,
    float& d_cc02,
    float& d_cc11,
    float& d_cc12,
    float& d_cc22
)
{
    // ----- Replicate forward culling -----
    if (mc_z < near_plane || mc_z > far_plane)
        return false;
    float mean2d_x, mean2d_y;
    if (!CameraOps<F>::project(mc_x, mc_y, mc_z, fx, fy, cx, cy, lens.dist, mean2d_x, mean2d_y))
        return false;
    const float rz = 1.0f / mc_z;
    LensJacobian<F> lj;
    if (!lj.compute(mc_x, mc_y, rz, fx, fy, lens))
        return false;
    const float* j = lj.j;

    float c2d00_raw, c2d01, c2d11_raw;
    device_lens_covar2d(j, cc00, cc01, cc02, cc11, cc12, cc22, c2d00_raw, c2d01, c2d11_raw);
    const float c2d00 = c2d00_raw + eps2d;
    const float c2d11 = c2d11_raw + eps2d;
    const float det = c2d00 * c2d11 - c2d01 * c2d01;
    if (det <= 0.0f)
        return false;
    if (opacity_cond < ALPHA_THRESHOLD)
        return false;
    const float extend = fminf(3.33f, sqrtf(2.0f * __logf(opacity_cond / ALPHA_THRESHOLD)));
    const int32_t radius_x = (int32_t)(ceilf(extend * sqrtf(c2d00)));
    const int32_t radius_y = (int32_t)(ceilf(extend * sqrtf(c2d11)));
    if (mean2d_x + (float)radius_x <= 0.0f || mean2d_x - (float)radius_x >= (float)image_width ||
        mean2d_y + (float)radius_y <= 0.0f || mean2d_y - (float)radius_y >= (float)image_height)
        return false;

    // ----- Conic and compensation backward (as device_persp_proj_ewa_bwd) -----
    const float inv_det = 1.0f / det;
    const float inv_det2 = inv_det * inv_det;
    float d_c2d00 = up_dc * inv_det;
    float d_c2d01 = -up_db * inv_det;
    float d_c2d11 = up_da * inv_det;
    const float d_det = -(up_da * c2d11 + up_db * (-c2d01) + up_dc * c2d00) * inv_det2;
    d_c2d00 += d_det * c2d11;
    d_c2d01 += d_det * (-2.0f * c2d01);
    d_c2d11 += d_det * c2d00;

    compensation = 1.0f;
    if (calc_compensations) {
        const float det_orig = c2d00_raw * c2d11_raw - c2d01 * c2d01;
        compensation = det_orig > 0.0f ? sqrtf(det_orig * inv_det) : 0.0f;
        if (det_orig > 0.0f) {
            const float d_comp = up_dopac * opacity_cond;
            const float d_det_orig = 0.5f * d_comp * compensation / det_orig;
            const float d_det_blur = -0.5f * d_comp * compensation * inv_det;
            d_c2d00 += d_det_orig * c2d11_raw + d_det_blur * c2d11;
            d_c2d11 += d_det_orig * c2d00_raw + d_det_blur * c2d00;
            d_c2d01 += -2.0f * c2d01 * (d_det_orig + d_det_blur);
        }
    }

    // ----- Σ2d = J Σ_c Jᵀ: G is the symmetric upstream (the off-diagonal variable stands for both entries) -----
    const float g00 = d_c2d00, g01 = 0.5f * d_c2d01, g11 = d_c2d11;
    // GJ = G · J (2×3)
    float gj[6];
    for (int k = 0; k < 3; ++k) {
        gj[k] = g00 * j[k] + g01 * j[3 + k];
        gj[3 + k] = g01 * j[k] + g11 * j[3 + k];
    }
    // v_Σc = Jᵀ G J, off-diagonals doubled onto the upper-tri variables
    d_cc00 += j[0] * gj[0] + j[3] * gj[3];
    d_cc01 += 2.f * (j[0] * gj[1] + j[3] * gj[4]);
    d_cc02 += 2.f * (j[0] * gj[2] + j[3] * gj[5]);
    d_cc11 += j[1] * gj[1] + j[4] * gj[4];
    d_cc12 += 2.f * (j[1] * gj[2] + j[4] * gj[5]);
    d_cc22 += j[2] * gj[2] + j[5] * gj[5];

    // v_J = 2 G J Σ_c, then V = diag(f) v_J
    float v[6];
    for (int i = 0; i < 2; ++i) {
        const float* r = gj + 3 * i;
        const float f2 = 2.f * (i == 0 ? fx : fy);
        v[3 * i + 0] = f2 * (r[0] * cc00 + r[1] * cc01 + r[2] * cc02);
        v[3 * i + 1] = f2 * (r[0] * cc01 + r[1] * cc11 + r[2] * cc12);
        v[3 * i + 2] = f2 * (r[0] * cc02 + r[1] * cc12 + r[2] * cc22);
    }

    // ----- Mean through the projection: J_m = diag(f) J_D(p) J_P(p), p unclamped -----
    const float px_u = mc_x * rz, py_u = mc_y * rz;
    float jd_m[4] = {lj.jd[0], lj.jd[1], lj.jd[2], lj.jd[3]};
    if (lj.clamped_x || lj.clamped_y) {
        float q1, q2;
        device_lens_jd<F>(px_u, py_u, lens.dist, jd_m, q1, q2); // u ≥ ū, inside the domain: project passed
    }
    {
        const float wx = up_dm2d_x * fx, wy = up_dm2d_y * fy; // diag(f)ᵀ v_m
        const float ax = wx * jd_m[0] + wy * jd_m[2];         // J_Dᵀ ·
        const float ay = wx * jd_m[1] + wy * jd_m[3];
        d_mc_x += ax * rz;
        d_mc_y += ay * rz;
        d_mc_z += -(ax * px_u + ay * py_u) * rz;
    }

    // ----- Mean through J (Hessian contraction), J_D and J_P at p̄ -----
    const float px = lj.px, py = lj.py;
    // A = V J_Pᵀ (2×2), J_P = [[rz, 0, −p̄x rz], [0, rz, −p̄y rz]]
    float a[4];
    for (int i = 0; i < 2; ++i) {
        a[2 * i + 0] = (v[3 * i + 0] - v[3 * i + 2] * px) * rz;
        a[2 * i + 1] = (v[3 * i + 1] - v[3 * i + 2] * py) * rz;
    }
    // Radial: g = 2ψ′ (p tr A + A p + Aᵀ p) + 4ψ″ (pᵀ A p) p
    const float tr = a[0] + a[3];
    const float apx = a[0] * px + a[1] * py, apy = a[2] * px + a[3] * py;
    const float atpx = a[0] * px + a[2] * py, atpy = a[1] * px + a[3] * py;
    const float pap = px * apx + py * apy;
    float gx = 2.f * lj.psi1 * (px * tr + apx + atpx) + 4.f * lj.psi2 * pap * px;
    float gy = 2.f * lj.psi1 * (py * tr + apy + atpy) + 4.f * lj.psi2 * pap * py;
    EwaLens<F>::add_tangential_contraction(px, py, lens.dist, a, gx, gy);
    // ∂p̄_k/∂x_c = (δ/z, −p_k/z) on an unclamped axis, 0 on a clamped one
    if (!lj.clamped_x) {
        d_mc_x += gx * rz;
        d_mc_z -= gx * px * rz;
    }
    if (!lj.clamped_y) {
        d_mc_y += gy * rz;
        d_mc_z -= gy * py * rz;
    }

    // B = J_Dᵀ V (2×3), contracted with ∂J_P/∂x_c
    const float* jd = lj.jd;
    const float b00 = jd[0] * v[0] + jd[2] * v[3];
    const float b02 = jd[0] * v[2] + jd[2] * v[5];
    const float b11 = jd[1] * v[1] + jd[3] * v[4];
    const float b12 = jd[1] * v[2] + jd[3] * v[5];
    const float rz2 = rz * rz;
    d_mc_z -= (b00 + b11) * rz2;
    if (lj.clamped_x) {
        d_mc_z += b02 * px * rz2;
    }
    else {
        d_mc_x -= b02 * rz2;
        d_mc_z += 2.f * b02 * px * rz2;
    }
    if (lj.clamped_y) {
        d_mc_z += b12 * py * rz2;
    }
    else {
        d_mc_y -= b12 * rz2;
        d_mc_z += 2.f * b12 * py * rz2;
    }
    return true;
}

/** The kernels' EWA projection: the pinhole device function when !kLens (dist unused), the lens path otherwise. */
template <ProjectionFamily F, bool kLens, typename... Rest>
__device__ __forceinline__ bool device_ewa_project(
    float mc_x,
    float mc_y,
    float mc_z,
    float cc00,
    float cc01,
    float cc02,
    float cc11,
    float cc12,
    float cc22,
    float fx,
    float fy,
    float cx,
    float cy,
    EwaLensCam lens,
    Rest&&... rest
)
{
    if constexpr (kLens)
        return device_lens_proj_ewa<
            F>(mc_x, mc_y, mc_z, cc00, cc01, cc02, cc11, cc12, cc22, fx, fy, cx, cy, lens, rest...);
    else
        return device_persp_proj_ewa(mc_x, mc_y, mc_z, cc00, cc01, cc02, cc11, cc12, cc22, fx, fy, cx, cy, rest...);
}

/** Backward counterpart of device_ewa_project. */
template <ProjectionFamily F, bool kLens, typename... Rest>
__device__ __forceinline__ bool device_ewa_project_bwd(
    float mc_x,
    float mc_y,
    float mc_z,
    float cc00,
    float cc01,
    float cc02,
    float cc11,
    float cc12,
    float cc22,
    float fx,
    float fy,
    float cx,
    float cy,
    EwaLensCam lens,
    Rest&&... rest
)
{
    if constexpr (kLens)
        return device_lens_proj_ewa_bwd<
            F>(mc_x, mc_y, mc_z, cc00, cc01, cc02, cc11, cc12, cc22, fx, fy, cx, cy, lens, rest...);
    else
        return device_persp_proj_ewa_bwd(mc_x, mc_y, mc_z, cc00, cc01, cc02, cc11, cc12, cc22, fx, fy, cx, cy, rest...);
}

/** That camera's lens; empty on the undistorted pinhole path. */
template <ProjectionFamily F, bool kLens>
__device__ __forceinline__ EwaLensCam device_ewa_lens(const float* dist, const float* lims, int cam_idx)
{
    if constexpr (!kLens)
        return {nullptr, nullptr};
    return {dist + cam_idx * CameraOps<F>::n_slots, lims != nullptr ? lims + cam_idx * 4 : nullptr};
}
