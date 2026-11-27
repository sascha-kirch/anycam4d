/**
 * Device helpers for rot_4d conditioning: SO(4) rotation, Schur blocks, conditioning.
 * Upper-tri covariance layout (as gsplat): [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22].
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "linalg.cuh"

/**
 * Guard on the marginal temporal variance in the Schur denominator and the temporal gate.
 * It also sets the narrowest renderable temporal extent, sqrt(eps) = 3.16e-3. The only CUDA
 * definition; must equal anycam4d/math/temporal.py. A double literal, cast at each use.
 */
#define TEMPORAL_VARIANCE_EPS 1e-5

/**
 * Dot product of two length-4 arrays.
 *
 * @param a  First array of 4 elements.
 * @param b  Second array of 4 elements.
 * @return   a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
 */
template <typename T> __device__ __forceinline__ T dot4(const T a[4], const T b[4])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/**
 * Sigmoid activation: σ(x) = 1 / (1 + exp(−x)).
 *
 * @param x  Input scalar.
 * @return   σ(x) in (0, 1).
 */
template <typename T> __device__ __forceinline__ T sigmoid(T x)
{
    return (T)1 / ((T)1 + exp(-x));
}

/**
 * Derivative of sigmoid w.r.t. its input: dσ/dx = σ(x) * (1 − σ(x)).
 *
 * Takes the already-computed sigmoid value to avoid recomputing exp().
 *
 * @param sig_x  σ(x), the forward sigmoid output.
 * @return       σ(x) * (1 − σ(x)).
 */
template <typename T> __device__ __forceinline__ T sigmoid_grad(T sig_x)
{
    return sig_x * ((T)1 - sig_x);
}

/**
 * Normalize a quaternion in-place: q ← q / ‖q‖.
 *
 * The original norm is returned so the backward pass can use it without recomputing a square root.
 *
 * @param q  Quaternion [w, x, y, z] to normalize (modified in place).
 * @return   ‖q‖ before normalization.
 */
template <typename T> __device__ __forceinline__ T quat_normalize(T q[4])
{
    T n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    T inv = (T)1 / n;
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
    return n;
}

/**
 * Backward of y = x / ‖x‖: d_x = (d_y - y · dot(y, d_y)) / ‖x‖.
 */
template <typename T> __device__ __forceinline__ void quat_normalize_bwd(const T y[4], const T d_y[4], T norm, T d_x[4])
{
    T y_dot_dy = dot4(y, d_y);
    T inv_norm = (T)1 / norm;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        d_x[i] = (d_y[i] - y[i] * y_dot_dy) * inv_norm;
    }
}

/**
 * SO(4) rotation A = M_l(ql) @ M_r(qr) from normalized wxyz quaternions ql = (a, b, c, d), qr = (p, q, r, s):
 * @code
 *   M_l = [[ a, -b, -c, -d],      M_r = [[ p,  q,  r,  s],
 *          [ b,  a, -d,  c],             [-q,  p, -s,  r],
 *          [ c,  d,  a, -b],             [-r,  s,  p, -q],
 *          [ d, -c,  b,  a]]             [-s, -r,  q,  p]]
 * @endcode
 * R_4d = A.flip([0, 1]), applied in build_sigma_schur.
 */
template <typename T>
__device__ __forceinline__ void build_A_rot4d(
    const T ql[4], // [a,b,c,d] wxyz, already normalized
    const T qr[4], // [p,q,r,s] wxyz, already normalized
    T A[4][4]
)
{
    const T a = ql[0], b = ql[1], c = ql[2], d = ql[3];
    const T p = qr[0], q = qr[1], r = qr[2], s = qr[3];

    A[0][0] = a * p + b * q + c * r + d * s;
    A[0][1] = a * q - b * p - c * s + d * r;
    A[0][2] = a * r + b * s - c * p - d * q;
    A[0][3] = a * s - b * r + c * q - d * p;

    A[1][0] = b * p - a * q + d * r - c * s;
    A[1][1] = b * q + a * p - d * s - c * r;
    A[1][2] = b * r - a * s - d * p + c * q;
    A[1][3] = b * s + a * r + d * q + c * p;

    A[2][0] = c * p - d * q - a * r + b * s;
    A[2][1] = c * q + d * p + a * s + b * r;
    A[2][2] = c * r - d * s + a * p - b * q;
    A[2][3] = c * s + d * r - a * q - b * p;

    A[3][0] = d * p + c * q - b * r - a * s;
    A[3][1] = d * q - c * p + b * s - a * r;
    A[3][2] = d * r + c * s + b * p + a * q;
    A[3][3] = d * s - c * r - b * q + a * p;
}

/**
 * Marginal Schur blocks of Σ_4d = L Lᵀ with L[i][k] = A[3-i][3-k] · exp(scales_4d_log[k]):
 * @code
 *   Σ_4d = | cov_3d_marg  cov_xt     |
 *          | cov_xtᵀ      cov_t_marg |
 * @endcode
 * cov_3d_marg (upper-tri) is not what the renderer draws; that is the conditional from
 * device_condition_rot4d. Same type and layout, so the suffix is the only guard.
 */
template <typename T>
__device__ __forceinline__ void build_sigma_schur(
    const T A[4][4],
    const T s[4],     // exp(scales_4d_log)
    T cov_3d_marg[6], // out: upper-tri of the marginal spatial block
    T cov_xt[3],      // out: space-time cross-covariance
    T& cov_t_marg     // out: marginal temporal variance
)
{
#pragma unroll
    for (int i = 0; i < 6; ++i)
        cov_3d_marg[i] = (T)0;
#pragma unroll
    for (int i = 0; i < 3; ++i)
        cov_xt[i] = (T)0;
    cov_t_marg = (T)0;

    // Column k of L: L[row][k] = R_4d[row][k] * s[k] = A[3-row][3-k] * s[k]
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        T Lcol[4];
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            Lcol[row] = A[3 - row][3 - k] * s[k];
        }

        // Accumulate the marginal spatial block (upper triangle, symmetric)
        cov_3d_marg[0] += Lcol[0] * Lcol[0]; // (0,0)
        cov_3d_marg[1] += Lcol[0] * Lcol[1]; // (0,1)
        cov_3d_marg[2] += Lcol[0] * Lcol[2]; // (0,2)
        cov_3d_marg[3] += Lcol[1] * Lcol[1]; // (1,1)
        cov_3d_marg[4] += Lcol[1] * Lcol[2]; // (1,2)
        cov_3d_marg[5] += Lcol[2] * Lcol[2]; // (2,2)

        cov_xt[0] += Lcol[0] * Lcol[3];
        cov_xt[1] += Lcol[1] * Lcol[3];
        cov_xt[2] += Lcol[2] * Lcol[3];

        cov_t_marg += Lcol[3] * Lcol[3];
    }
}

/**
 * Backward of build_sigma_schur to d_A (written) and d_scales_4d_log (accumulated, through exp).
 */
template <typename T>
__device__ __forceinline__ void build_sigma_schur_bwd(
    const T A[4][4],
    const T s[4],
    const T d_cov_3d_marg[6],
    const T d_cov_xt[3],
    T d_cov_t_marg, // scalar, pass by value
    T d_A[4][4],
    T d_scales_4d_log[4] // accumulate into existing d_scales_4d_log
)
{
    // Zero d_A
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            d_A[i][j] = (T)0;

            // Replay column-by-column accumulation and compute d_Lcol
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        T Lcol[4];
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            Lcol[row] = A[3 - row][3 - k] * s[k];
        }

        // d_Lcol contributions from d_cov_3d_marg, d_cov_xt, d_cov_t_marg

        // From cov_3d_marg: cov_3d_marg[triu] += Lcol[i]*Lcol[j]
        // d_Lcol[i] += Σ_{j} (d_cov_3d_marg_full[i][j] + d_cov_3d_marg_full[j][i]) * Lcol[j]
        //            = 2 * Σ_{j in 0..2} d_cov_3d_marg_sym[i][j] * Lcol[j]
        // But d_cov_3d_marg is upper-tri only (no double-count needed if we use symmetry):
        //   d_cov_3d_marg[0](0,0) contrib: dL[0]+=2*d_cov_3d_marg[0]*L[0]
        //   d_cov_3d_marg[1](0,1) contrib: dL[0]+=d_cov_3d_marg[1]*L[1], dL[1]+=d_cov_3d_marg[1]*L[0]
        //   etc. — effectively d_Lcol = 2*(d_cov_3d_marg_sym @ Lcol[:3]) for entries 0..2

        T dLcol[4] = {(T)0, (T)0, (T)0, (T)0};

        // Symmetric 3x3 gradient (upper-tri stored, so factor 2 on off-diag)
        dLcol[0] += (T)2 * d_cov_3d_marg[0] * Lcol[0] + d_cov_3d_marg[1] * Lcol[1] + d_cov_3d_marg[2] * Lcol[2];
        dLcol[1] += d_cov_3d_marg[1] * Lcol[0] + (T)2 * d_cov_3d_marg[3] * Lcol[1] + d_cov_3d_marg[4] * Lcol[2];
        dLcol[2] += d_cov_3d_marg[2] * Lcol[0] + d_cov_3d_marg[4] * Lcol[1] + (T)2 * d_cov_3d_marg[5] * Lcol[2];

        // From cov_xt: cov_xt[i] += Lcol[i]*Lcol[3]
        dLcol[0] += d_cov_xt[0] * Lcol[3];
        dLcol[1] += d_cov_xt[1] * Lcol[3];
        dLcol[2] += d_cov_xt[2] * Lcol[3];
        dLcol[3] += d_cov_xt[0] * Lcol[0] + d_cov_xt[1] * Lcol[1] + d_cov_xt[2] * Lcol[2];

        // From cov_t_marg: cov_t_marg += Lcol[3]^2
        dLcol[3] += (T)2 * d_cov_t_marg * Lcol[3];

        // d_A and d_s from Lcol[row] = A[3-row][3-k] * s[k]
        //   d_A[3-row][3-k] += dLcol[row] * s[k]
        //   d_s[k]          += Σ_row dLcol[row] * A[3-row][3-k]
        T d_sk = (T)0;
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            d_A[3 - row][3 - k] += dLcol[row] * s[k];
            d_sk += dLcol[row] * A[3 - row][3 - k];
        }
        // Chain through s[k] = exp(scales_4d_log[k]):  d_scales_4d_log[k] += d_s[k] * s[k]
        d_scales_4d_log[k] += d_sk * s[k];
    }
}

/**
 * Backward of A = M_l(ql) @ M_r(qr): d_M_l = d_A @ M_rᵀ and d_M_r = M_lᵀ @ d_A, read off
 * entry-wise via the fixed sign patterns of M_l and M_r.
 */
template <typename T>
__device__ __forceinline__ void build_A_rot4d_bwd(
    const T ql[4], // normalized
    const T qr[4], // normalized
    const T d_A[4][4],
    T d_ql_hat[4], // out: gradient w.r.t. normalized ql
    T d_qr_hat[4]  // out: gradient w.r.t. normalized qr
)
{
    const T a = ql[0], b = ql[1], c = ql[2], d = ql[3];
    const T p = qr[0], q = qr[1], r = qr[2], s = qr[3];

    // d_ql_hat from d_M_l = d_A @ M_rᵀ; each component sums the entries where it appears in M_l.

    d_ql_hat[0] = (T)0;
    d_ql_hat[1] = (T)0;
    d_ql_hat[2] = (T)0;
    d_ql_hat[3] = (T)0;

    // Signed positions of each component of q_l in M_l (rows [a,-b,-c,-d], [b,a,-d,c], [c,d,a,-b], [d,-c,b,a]):
    //   a: +1 on the diagonal
    //   b: (-1 at (0,1)), (+1 at (1,0)), (-1 at (2,3)), (+1 at (3,2))
    //   c: positions (-1 at (0,2)), (+1 at (1,3)), (+1 at (2,0)), (-1 at (3,1))
    //   d: positions (-1 at (0,3)), (-1 at (1,2)), (+1 at (2,1)), (+1 at (3,0))

    // d_M_l[i][j] = (d_A @ M_r^T)[i][j] = Σ_l d_A[i][l] * M_r[j][l]

    // For d_a:
    // d_a = d_M_l[0][0] + d_M_l[1][1] + d_M_l[2][2] + d_M_l[3][3]
    // d_M_l[0][0] = Σ_l d_A[0][l] * M_r[0][l] = d_A[0] · M_r_row0
    // d_M_l[1][1] = Σ_l d_A[1][l] * M_r[1][l] = d_A[1] · M_r_row1
    // d_M_l[2][2] = Σ_l d_A[2][l] * M_r[2][l] = d_A[2] · M_r_row2
    // d_M_l[3][3] = Σ_l d_A[3][l] * M_r[3][l] = d_A[3] · M_r_row3

    // M_r rows: row0=(p,q,r,s), row1=(-q,p,-s,r), row2=(-r,s,p,-q),
    // row3=(-s,-r,q,p)
    d_ql_hat[0] += d_A[0][0] * p + d_A[0][1] * q + d_A[0][2] * r + d_A[0][3] * s          // d_M_l[0][0]
                   + d_A[1][0] * (-q) + d_A[1][1] * p + d_A[1][2] * (-s) + d_A[1][3] * r  // d_M_l[1][1]
                   + d_A[2][0] * (-r) + d_A[2][1] * s + d_A[2][2] * p + d_A[2][3] * (-q)  // d_M_l[2][2]
                   + d_A[3][0] * (-s) + d_A[3][1] * (-r) + d_A[3][2] * q + d_A[3][3] * p; // d_M_l[3][3]

    // For d_b:
    // d_b = -d_M_l[0][1] + d_M_l[1][0] - d_M_l[2][3] + d_M_l[3][2]
    // d_M_l[0][1] = d_A[0] · M_r_row1 = d_A[0]·(-q,p,-s,r)
    // d_M_l[1][0] = d_A[1] · M_r_row0 = d_A[1]·(p,q,r,s)
    // d_M_l[2][3] = d_A[2] · M_r_row3 = d_A[2]·(-s,-r,q,p)
    // d_M_l[3][2] = d_A[3] · M_r_row2 = d_A[3]·(-r,s,p,-q)
    d_ql_hat[1] += -(d_A[0][0] * (-q) + d_A[0][1] * p + d_A[0][2] * (-s) + d_A[0][3] * r)   // -d_M_l[0][1]
                   + (d_A[1][0] * p + d_A[1][1] * q + d_A[1][2] * r + d_A[1][3] * s)        // +d_M_l[1][0]
                   - (d_A[2][0] * (-s) + d_A[2][1] * (-r) + d_A[2][2] * q + d_A[2][3] * p)  // -d_M_l[2][3]
                   + (d_A[3][0] * (-r) + d_A[3][1] * s + d_A[3][2] * p + d_A[3][3] * (-q)); // +d_M_l[3][2]

    // For d_c:
    // d_c = -d_M_l[0][2] + d_M_l[1][3] + d_M_l[2][0] - d_M_l[3][1]
    // d_M_l[0][2] = d_A[0] · M_r_row2 = d_A[0]·(-r,s,p,-q)
    // d_M_l[1][3] = d_A[1] · M_r_row3 = d_A[1]·(-s,-r,q,p)
    // d_M_l[2][0] = d_A[2] · M_r_row0 = d_A[2]·(p,q,r,s)
    // d_M_l[3][1] = d_A[3] · M_r_row1 = d_A[3]·(-q,p,-s,r)
    d_ql_hat[2] += -(d_A[0][0] * (-r) + d_A[0][1] * s + d_A[0][2] * p + d_A[0][3] * (-q))   // -d_M_l[0][2]
                   + (d_A[1][0] * (-s) + d_A[1][1] * (-r) + d_A[1][2] * q + d_A[1][3] * p)  // +d_M_l[1][3]
                   + (d_A[2][0] * p + d_A[2][1] * q + d_A[2][2] * r + d_A[2][3] * s)        // +d_M_l[2][0]
                   - (d_A[3][0] * (-q) + d_A[3][1] * p + d_A[3][2] * (-s) + d_A[3][3] * r); // -d_M_l[3][1]

    // For d_d:
    // d_d = -d_M_l[0][3] - d_M_l[1][2] + d_M_l[2][1] + d_M_l[3][0]
    // d_M_l[0][3] = d_A[0] · M_r_row3 = d_A[0]·(-s,-r,q,p)
    // d_M_l[1][2] = d_A[1] · M_r_row2 = d_A[1]·(-r,s,p,-q)
    // d_M_l[2][1] = d_A[2] · M_r_row1 = d_A[2]·(-q,p,-s,r)
    // d_M_l[3][0] = d_A[3] · M_r_row0 = d_A[3]·(p,q,r,s)
    d_ql_hat[3] += -(d_A[0][0] * (-s) + d_A[0][1] * (-r) + d_A[0][2] * q + d_A[0][3] * p)  // -d_M_l[0][3]
                   - (d_A[1][0] * (-r) + d_A[1][1] * s + d_A[1][2] * p + d_A[1][3] * (-q)) // -d_M_l[1][2]
                   + (d_A[2][0] * (-q) + d_A[2][1] * p + d_A[2][2] * (-s) + d_A[2][3] * r) // +d_M_l[2][1]
                   + (d_A[3][0] * p + d_A[3][1] * q + d_A[3][2] * r + d_A[3][3] * s);      // +d_M_l[3][0]

    // d_qr_hat from d_M_r = M_lᵀ @ d_A; each component sums the entries where it appears in M_r.
    d_qr_hat[0] = (d_A[0][0] * a + d_A[1][0] * b + d_A[2][0] * c + d_A[3][0] * d) +
                  (d_A[0][1] * (-b) + d_A[1][1] * a + d_A[2][1] * d + d_A[3][1] * (-c)) +
                  (d_A[0][2] * (-c) + d_A[1][2] * (-d) + d_A[2][2] * a + d_A[3][2] * b) +
                  (d_A[0][3] * (-d) + d_A[1][3] * c + d_A[2][3] * (-b) + d_A[3][3] * a);

    // For d_q: M_r positions with q: (0,1)=+q, (1,0)=-q, (2,3)=-q, (3,2)=+q
    //   d_q = d_M_r[0][1] - d_M_r[1][0] - d_M_r[2][3] + d_M_r[3][2]
    //   d_M_r[0][1] = M_l_col0 · d_A[:,1] = (a,b,c,d)·d_A[:,1]
    //   d_M_r[1][0] = M_l_col1 · d_A[:,0] = (-b,a,d,-c)·d_A[:,0]
    //   d_M_r[2][3] = M_l_col2 · d_A[:,3] = (-c,-d,a,b)·d_A[:,3]
    //   d_M_r[3][2] = M_l_col3 · d_A[:,2] = (-d,c,-b,a)·d_A[:,2]
    d_qr_hat[1] = (d_A[0][1] * a + d_A[1][1] * b + d_A[2][1] * c + d_A[3][1] * d) -
                  (d_A[0][0] * (-b) + d_A[1][0] * a + d_A[2][0] * d + d_A[3][0] * (-c)) -
                  (d_A[0][3] * (-c) + d_A[1][3] * (-d) + d_A[2][3] * a + d_A[3][3] * b) +
                  (d_A[0][2] * (-d) + d_A[1][2] * c + d_A[2][2] * (-b) + d_A[3][2] * a);

    // For d_r: M_r positions with r: (0,2)=+r, (1,3)=+r, (2,0)=-r, (3,1)=-r
    //   d_r = d_M_r[0][2] + d_M_r[1][3] - d_M_r[2][0] - d_M_r[3][1]
    //   d_M_r[0][2] = M_l_col0 · d_A[:,2] = (a,b,c,d)·d_A[:,2]
    //   d_M_r[1][3] = M_l_col1 · d_A[:,3] = (-b,a,d,-c)·d_A[:,3]
    //   d_M_r[2][0] = M_l_col2 · d_A[:,0] = (-c,-d,a,b)·d_A[:,0]
    //   d_M_r[3][1] = M_l_col3 · d_A[:,1] = (-d,c,-b,a)·d_A[:,1]
    d_qr_hat[2] = (d_A[0][2] * a + d_A[1][2] * b + d_A[2][2] * c + d_A[3][2] * d) +
                  (d_A[0][3] * (-b) + d_A[1][3] * a + d_A[2][3] * d + d_A[3][3] * (-c)) -
                  (d_A[0][0] * (-c) + d_A[1][0] * (-d) + d_A[2][0] * a + d_A[3][0] * b) -
                  (d_A[0][1] * (-d) + d_A[1][1] * c + d_A[2][1] * (-b) + d_A[3][1] * a);

    // For d_s: M_r positions with s: (0,3)=+s, (1,2)=-s, (2,1)=+s, (3,0)=-s
    //   d_s = d_M_r[0][3] - d_M_r[1][2] + d_M_r[2][1] - d_M_r[3][0]
    //   d_M_r[0][3] = M_l_col0 · d_A[:,3] = (a,b,c,d)·d_A[:,3]
    //   d_M_r[1][2] = M_l_col1 · d_A[:,2] = (-b,a,d,-c)·d_A[:,2]
    //   d_M_r[2][1] = M_l_col2 · d_A[:,1] = (-c,-d,a,b)·d_A[:,1]
    //   d_M_r[3][0] = M_l_col3 · d_A[:,0] = (-d,c,-b,a)·d_A[:,0]
    d_qr_hat[3] = (d_A[0][3] * a + d_A[1][3] * b + d_A[2][3] * c + d_A[3][3] * d) -
                  (d_A[0][2] * (-b) + d_A[1][2] * a + d_A[2][2] * d + d_A[3][2] * (-c)) +
                  (d_A[0][1] * (-c) + d_A[1][1] * (-d) + d_A[2][1] * a + d_A[3][1] * b) -
                  (d_A[0][0] * (-d) + d_A[1][0] * c + d_A[2][0] * (-b) + d_A[3][0] * a);
}

/**
 * Factored conditional covariance: with M = R₄ diag(s) (Σ₄ = M Mᵀ, rows x | t = m), u = Σ_xt = M_x m and
 * k = 1 / ((|m|² + TE)(1 + r)), r = √(TE / (|m|² + TE)), N = M_x − k u mᵀ gives N Nᵀ = Σ_xx − u uᵀ / (Σ_tt + TE)
 * exactly, without the subtraction that cancels in fp32 for fast, thin Gaussians.
 */
template <typename T> struct ConditionedN {
    T N[3][4];
    T m[4];
    T k, r;
};

template <typename T>
__device__ __forceinline__ ConditionedN<T>
device_conditioned_N(const T A[4][4], const T s[4], const T cov_xt[3], T cov_t_marg)
{
    ConditionedN<T> c;
    const T D = cov_t_marg + (T)TEMPORAL_VARIANCE_EPS;
    c.r = sqrt((T)TEMPORAL_VARIANCE_EPS / D);
    c.k = (T)1 / (D * ((T)1 + c.r));
#pragma unroll
    for (int col = 0; col < 4; ++col) {
        c.m[col] = A[0][3 - col] * s[col]; // M[3][col]
#pragma unroll
        for (int row = 0; row < 3; ++row)
            c.N[row][col] = A[3 - row][3 - col] * s[col] - c.k * cov_xt[row] * c.m[col];
    }
    return c;
}

/** Packed lower L with L Lᵀ = N Nᵀ + cov_eps·I, by QR of [Nᵀ; √cov_eps·I] (7×3). */
template <typename T> __device__ __forceinline__ void device_factor_of_N(const T N[3][4], T cov_eps, T L_out[6])
{
    T Q[7][3];
#pragma unroll
    for (int col = 0; col < 4; ++col)
#pragma unroll
        for (int row = 0; row < 3; ++row)
            Q[col][row] = N[row][col];
    const T se = sqrt(cov_eps);
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            Q[4 + i][j] = i == j ? se : (T)0;
    device_lower_factor_qr<T, 7>(Q, L_out);
}

/** Packed lower factor of the conditional covariance + cov_eps·I. */
template <typename T>
__device__ __forceinline__ void
device_conditioned_factor(const T A[4][4], const T s[4], const T cov_xt[3], T cov_t_marg, T cov_eps, T L_out[6])
{
    const ConditionedN<T> c = device_conditioned_N(A, s, cov_xt, cov_t_marg);
    device_factor_of_N(c.N, cov_eps, L_out);
}

/**
 * N̄ → M̄ → d_A and d_scales_4d_log (both accumulated). M̄_x = N̄ − k v mᵀ and
 * m̄ = −k Nᵀv − k N̄ᵀu + (1 + r) k² (u·v) m with v = N̄ m, the derivative of N = M_x − k (M_x m) mᵀ.
 */
template <typename T>
__device__ __forceinline__ void device_conditioned_N_bwd(
    const T A[4][4],
    const T s[4],
    const ConditionedN<T>& c,
    const T cov_xt[3],
    const T Nbar[3][4],
    T d_A[4][4],
    T d_scales_4d_log[4]
)
{
    T v[3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
        v[i] = Nbar[i][0] * c.m[0] + Nbar[i][1] * c.m[1] + Nbar[i][2] * c.m[2] + Nbar[i][3] * c.m[3];
    const T uv = cov_xt[0] * v[0] + cov_xt[1] * v[1] + cov_xt[2] * v[2];
    const T km = ((T)1 + c.r) * c.k * c.k * uv;
#pragma unroll
    for (int col = 0; col < 4; ++col) {
        T Mbar[4];
#pragma unroll
        for (int row = 0; row < 3; ++row)
            Mbar[row] = Nbar[row][col] - c.k * v[row] * c.m[col];
        Mbar[3] = km * c.m[col];
#pragma unroll
        for (int row = 0; row < 3; ++row)
            Mbar[3] -= c.k * (c.N[row][col] * v[row] + Nbar[row][col] * cov_xt[row]);
        // M[row][col] = A[3-row][3-col] · s[col]
        T d_s = (T)0;
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            d_A[3 - row][3 - col] += Mbar[row] * s[col];
            d_s += Mbar[row] * A[3 - row][3 - col];
        }
        d_scales_4d_log[col] += d_s * s[col];
    }
}

/**
 * Backward of the packed factor L (from device_conditioned_factor) into d_A and d_scales_4d_log (accumulated):
 * N̄ = L⁻ᵀ (Φ + Φᵀ) L⁻¹ N, Φ = lower(Lᵀ L̄) with halved diagonal — the QR backward, with L⁻¹ N bounded.
 */
template <typename T>
__device__ __forceinline__ void device_conditioned_factor_bwd(
    const T A[4][4],
    const T s[4],
    const T cov_xt[3],
    T cov_t_marg,
    T cov_eps,
    const T dL[6],
    T d_A[4][4],
    T d_scales_4d_log[4]
)
{
    const ConditionedN<T> c = device_conditioned_N(A, s, cov_xt, cov_t_marg);
    T L[6];
    device_factor_of_N(c.N, cov_eps, L);
    T Lm[3][3], dLm[3][3];
    device_unpack_lower(L, Lm);
    device_unpack_lower(dL, dLm);
    T Psi[3][3]; // Φ + Φᵀ
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j <= i; ++j) {
            T acc = (T)0;
#pragma unroll
            for (int q = 0; q < 3; ++q)
                acc += Lm[q][i] * dLm[q][j];
            Psi[i][j] = acc;
            Psi[j][i] = acc;
        }
    T Nbar[3][4];
#pragma unroll
    for (int col = 0; col < 4; ++col) {
        const T n[3] = {c.N[0][col], c.N[1][col], c.N[2][col]};
        T y[3], z[3], x[3];
        device_solve_lower3x3(L, n, y);
#pragma unroll
        for (int i = 0; i < 3; ++i)
            z[i] = Psi[i][0] * y[0] + Psi[i][1] * y[1] + Psi[i][2] * y[2];
        device_solve_upper3x3(L, z, x);
#pragma unroll
        for (int i = 0; i < 3; ++i)
            Nbar[i][col] = x[i];
    }
    device_conditioned_N_bwd(A, s, c, cov_xt, Nbar, d_A, d_scales_4d_log);
}

/**
 * Backward of the conditional covariance from ∂/∂Σ (upper-tri, each stored off-diagonal standing for both entries)
 * into d_A and d_scales_4d_log (accumulated): N̄ = (G + Gᵀ) N.
 */
template <typename T>
__device__ __forceinline__ void device_conditioned_cov_bwd(
    const T A[4][4],
    const T s[4],
    const T cov_xt[3],
    T cov_t_marg,
    const T d_sigma[6],
    T d_A[4][4],
    T d_scales_4d_log[4]
)
{
    const ConditionedN<T> c = device_conditioned_N(A, s, cov_xt, cov_t_marg);
    const T S[3][3] = {
        {(T)2 * d_sigma[0], d_sigma[1], d_sigma[2]},
        {d_sigma[1], (T)2 * d_sigma[3], d_sigma[4]},
        {d_sigma[2], d_sigma[4], (T)2 * d_sigma[5]},
    };
    T Nbar[3][4];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int col = 0; col < 4; ++col)
            Nbar[i][col] = S[i][0] * c.N[0][col] + S[i][1] * c.N[1][col] + S[i][2] * c.N[2][col];
    device_conditioned_N_bwd(A, s, c, cov_xt, Nbar, d_A, d_scales_4d_log);
}

/**
 * Condition one 4D Gaussian at tau:
 *   mean   = means + cov_xt / cov_t_marg_safe · (tau - mu_t)
 *   L      = factor of cov_3d_marg - cov_xt cov_xtᵀ / cov_t_marg_safe + cov_eps·I   (packed lower, see above)
 *   w_t    = exp(-½ (tau - mu_t)² / cov_t_marg_safe)
 *   opac   = sigmoid(raw_opacity) · w_t, zeroed when w_t < prefilter_thr (0 disables)
 * Optionally the velocity v = cov_xt / cov_t_marg_safe (mean(τ') = mean(τ) + v (τ' − τ)) and 1 / cov_t_marg_safe.
 * Shared by every fused kernel.
 */
template <typename T>
__device__ __forceinline__ void device_condition_rot4d(
    const T* __restrict__ ql_ptr,
    const T* __restrict__ qr_ptr,
    const T* __restrict__ s4d_ptr,
    T m0,
    T m1,
    T m2,
    T mu_t_val,
    T raw_opacity,
    T tau,
    T cov_eps,
    T prefilter_thr,
    T& out_mx,
    T& out_my,
    T& out_mz,
    T out_L[6],
    T& out_wt,
    T& out_opacity_cond,
    T* out_v = nullptr,
    T* out_inv_cov_t_marg = nullptr
)
{
    T ql[4], qr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        ql[i] = ql_ptr[i];
        qr[i] = qr_ptr[i];
    }
    quat_normalize(ql);
    quat_normalize(qr);

    T s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        s[i] = exp(s4d_ptr[i]);

    T A[4][4];
    build_A_rot4d(ql, qr, A);

    T cov_3d_marg[6], cov_xt[3], cov_t_marg;
    build_sigma_schur(A, s, cov_3d_marg, cov_xt, cov_t_marg);

    const T dt = tau - mu_t_val;
    const T inv_cov_t_marg = (T)1 / (cov_t_marg + (T)TEMPORAL_VARIANCE_EPS);

    device_conditioned_factor(A, s, cov_xt, cov_t_marg, cov_eps, out_L);

    const T mean_corr_scale = dt * inv_cov_t_marg;
    out_mx = m0 + cov_xt[0] * mean_corr_scale;
    out_my = m1 + cov_xt[1] * mean_corr_scale;
    out_mz = m2 + cov_xt[2] * mean_corr_scale;

    out_wt = exp((T)-0.5 * dt * dt * inv_cov_t_marg);
    const T opacity_base = sigmoid(raw_opacity);
    const bool passes = (prefilter_thr <= (T)0) || (out_wt >= prefilter_thr);
    out_opacity_cond = passes ? opacity_base * out_wt : (T)0;
    if (out_v != nullptr) {
#pragma unroll
        for (int i = 0; i < 3; ++i)
            out_v[i] = cov_xt[i] * inv_cov_t_marg;
        *out_inv_cov_t_marg = inv_cov_t_marg;
    }
}
