/**
 * linalg.cuh — Family-agnostic 3×3 symmetric-matrix primitives.
 *
 * Generic linear-algebra device functions shared across covariance-handling
 * kernels: the UT projection (ut_4d.cuh) and the world-space rasterizer
 * (rasterize_from_world.cuh). These primitives carry no
 * feature-specific machinery (no sigma points, no camera models), so a kernel
 * can include this header without dragging in the rest of a feature's code.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * FMA 2×2 determinant ad − bc, evaluated as fma(a, d, −b·c) to keep one rounding
 * on the product difference (well-conditioned when ad and bc are same-sign).
 */
__device__ __forceinline__ float device_fma_det2(float a, float b, float c, float d)
{
    return fmaf(a, d, -b * c);
}

// ─────────────────────────────────────────────────────────────────────────────
// Packed lower-triangular factor L = [l00, l10, l20, l11, l21, l22], Σ = L Lᵀ
// ─────────────────────────────────────────────────────────────────────────────
// Covariances travel between kernels as this factor, never as Σ: re-factoring an fp32 Σ loses its thin axis.
// Every factor has a positive diagonal (conditioning adds cov_eps > 0), so the triangular solves never divide by zero.

/**
 * Lower factor of AᵀA (diagonal > 0) by Householder QR of the ROWS×3 matrix A (overwritten): A = QR, L = Rᵀ.
 * Backward stable, so unlike Cholesky of the formed Gram it keeps the thin axis of an ill-conditioned Σ.
 */
template <typename T, int ROWS> __device__ __forceinline__ void device_lower_factor_qr(T A[ROWS][3], T L_out[6])
{
    T R[3][3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        T nx2 = (T)0;
#pragma unroll
        for (int i = k; i < ROWS; ++i)
            nx2 += A[i][k] * A[i][k];
        const T nx = sqrt(nx2);
        // Reflect x onto -sign(x0)·‖x‖·e1: v = x - alpha·e1 has no cancellation; vᵀv = 2‖x‖(‖x‖ + |x0|).
        const T alpha = A[k][k] >= (T)0 ? -nx : nx;
        const T vtv = (T)2 * nx * (nx + fabs(A[k][k]));
        const T beta = vtv > (T)0 ? (T)2 / vtv : (T)0;
        A[k][k] -= alpha;
#pragma unroll
        for (int j = k + 1; j < 3; ++j) {
            T vy = (T)0;
#pragma unroll
            for (int i = k; i < ROWS; ++i)
                vy += A[i][k] * A[i][j];
            const T f = beta * vy;
#pragma unroll
            for (int i = k; i < ROWS; ++i)
                A[i][j] -= f * A[i][k];
        }
        // Row k of R, its sign flipped so that R_kk = ‖x‖ > 0 (R Rᵀ is unchanged).
        const T sgn = alpha < (T)0 ? (T)-1 : (T)1;
        R[k][k] = sgn * alpha;
#pragma unroll
        for (int j = k + 1; j < 3; ++j)
            R[k][j] = sgn * A[k][j];
    }
    L_out[0] = R[0][0];
    L_out[1] = R[0][1];
    L_out[2] = R[0][2];
    L_out[3] = R[1][1];
    L_out[4] = R[1][2];
    L_out[5] = R[2][2];
}

/** Unpack L into a row-major 3×3 matrix. */
template <typename T> __device__ __forceinline__ void device_unpack_lower(const T L[6], T M[3][3])
{
    M[0][0] = L[0];
    M[0][1] = (T)0;
    M[0][2] = (T)0;
    M[1][0] = L[1];
    M[1][1] = L[3];
    M[1][2] = (T)0;
    M[2][0] = L[2];
    M[2][1] = L[4];
    M[2][2] = L[5];
}

/** Upper-tri Σ = L Lᵀ, for consumers that need the covariance itself. */
template <typename T> __device__ __forceinline__ void device_gram_lower(const T L[6], T sigma[6])
{
    sigma[0] = L[0] * L[0];
    sigma[1] = L[0] * L[1];
    sigma[2] = L[0] * L[2];
    sigma[3] = L[1] * L[1] + L[3] * L[3];
    sigma[4] = L[1] * L[2] + L[3] * L[4];
    sigma[5] = L[2] * L[2] + L[4] * L[4] + L[5] * L[5];
}

/** Upper-tri (R L)(R L)ᵀ = R Σ Rᵀ, PSD as a product (R row-major). */
__device__ __forceinline__ void device_factor_congruence(const float R[3][3], const float L[6], float out[6])
{
    float B[3][3]; // B = R L
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        B[i][0] = R[i][0] * L[0] + R[i][1] * L[1] + R[i][2] * L[2];
        B[i][1] = R[i][1] * L[3] + R[i][2] * L[4];
        B[i][2] = R[i][2] * L[5];
    }
    out[0] = B[0][0] * B[0][0] + B[0][1] * B[0][1] + B[0][2] * B[0][2];
    out[1] = B[0][0] * B[1][0] + B[0][1] * B[1][1] + B[0][2] * B[1][2];
    out[2] = B[0][0] * B[2][0] + B[0][1] * B[2][1] + B[0][2] * B[2][2];
    out[3] = B[1][0] * B[1][0] + B[1][1] * B[1][1] + B[1][2] * B[1][2];
    out[4] = B[1][0] * B[2][0] + B[1][1] * B[2][1] + B[1][2] * B[2][2];
    out[5] = B[2][0] * B[2][0] + B[2][1] * B[2][1] + B[2][2] * B[2][2];
}

/**
 * Forward substitution: solve the lower-triangular system L y = b.
 *
 * @param L  Packed lower-tri factor [l00, l10, l20, l11, l21, l22].
 * @param b  Right-hand side [3].
 * @param y  Output solution [3].
 */
template <typename T> __device__ __forceinline__ void device_solve_lower3x3(const T L[6], const T b[3], T y[3])
{
    y[0] = b[0] / L[0];
    y[1] = (b[1] - L[1] * y[0]) / L[3];
    y[2] = (b[2] - L[2] * y[0] - L[4] * y[1]) / L[5];
}

/**
 * Back substitution: solve the upper-triangular system Lᵀ x = y.
 *
 * @param L  Packed lower-tri factor [l00, l10, l20, l11, l21, l22].
 * @param y  Right-hand side [3].
 * @param x  Output solution [3].
 */
template <typename T> __device__ __forceinline__ void device_solve_upper3x3(const T L[6], const T y[3], T x[3])
{
    x[2] = y[2] / L[5];
    x[1] = (y[1] - L[4] * x[2]) / L[3];
    x[0] = (y[0] - L[1] * x[1] - L[2] * x[2]) / L[0];
}
