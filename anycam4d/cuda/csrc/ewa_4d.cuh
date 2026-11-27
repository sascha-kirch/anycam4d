/**
 * EWA perspective projection and culling on raw floats, mirroring gsplat's ProjectionEWA3DGSFused.cu.
 * A culled Gaussian gets radii = 0 and occupies no tile, so its other outputs are never read.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// Alpha threshold below which a Gaussian is considered invisible (matches gsplat)
#ifndef ALPHA_THRESHOLD
#define ALPHA_THRESHOLD (1.f / 255.f)
#endif

/**
 * Project one camera-space Gaussian to screen space (EWA, pinhole): near/far clip, clamped Jacobian,
 * covar2d = J Σ_c Jᵀ + eps2d·I, PD check, opacity cull, opacity-aware radii, frustum cull, conic.
 * Covariance args are upper-tri. With calc_compensations, out_compensation = sqrt(det Σ2d_raw / det Σ2d_blurred),
 * else 1. Returns false when culled; the caller writes radii = 0.
 */
__device__ __forceinline__ bool device_persp_proj_ewa(
    // Camera-space mean
    float mc_x,
    float mc_y,
    float mc_z,
    // Camera-space covariance (upper-tri)
    float cc00,
    float cc01,
    float cc02,
    float cc11,
    float cc12,
    float cc22,
    // Camera intrinsics
    float fx,
    float fy,
    float cx,
    float cy,
    // EWA parameters
    float eps2d,
    float opacity_cond,
    float near_plane,
    float far_plane,
    int image_width,
    int image_height,
    bool calc_compensations,
    // Outputs
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
    // --- 1. Depth clip ---
    if (mc_z < near_plane || mc_z > far_plane)
        return false;

    out_depth = mc_z;

    const float rz = 1.0f / mc_z;
    const float rz2 = rz * rz;

    // --- 2. Clamped perspective Jacobian (matches gsplat's persp_proj) ---
    // Clamp the normalised coordinates to slightly outside the frustum to avoid
    // the covariance degenerating when a Gaussian centre is near the image edge.
    const float tan_fovx = 0.5f * (float)image_width / fx;
    const float tan_fovy = 0.5f * (float)image_height / fy;
    const float lim_x_pos = ((float)image_width - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = ((float)image_height - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    const float nx = mc_x * rz;
    const float ny = mc_y * rz;
    const float tx_clamped = mc_z * fminf(lim_x_pos, fmaxf(-lim_x_neg, nx));
    const float ty_clamped = mc_z * fminf(lim_y_pos, fmaxf(-lim_y_neg, ny));

    // J = [[fx/z,  0,     -fx·tx/z²],
    //      [0,     fy/z,  -fy·ty/z²]]
    const float j00 = fx * rz;
    const float j02 = -fx * tx_clamped * rz2;
    const float j11 = fy * rz;
    const float j12 = -fy * ty_clamped * rz2;

    // --- 3 & 4. 2D covariance: covar2d = J Σ_c Jᵀ + eps2d·I ---
    // Intermediate: JC = J @ Σ_c  (2×3, but only [0,1] rows × 3 cols needed)
    // Row 0 of JC: jc0k = j00·Σ_c[0,k] + j02·Σ_c[2,k]
    // Row 1 of JC: jc1k = j11·Σ_c[1,k] + j12·Σ_c[2,k]
    //   (we only need k=0,1,2 to compute the 2×2 product — and k=1,2 for row 1)
    const float jc00 = j00 * cc00 + j02 * cc02;
    const float jc01 = j00 * cc01 + j02 * cc12;
    const float jc02 = j00 * cc02 + j02 * cc22;
    const float jc11 = j11 * cc11 + j12 * cc12;
    const float jc12 = j11 * cc12 + j12 * cc22;

    // covar2d = JC @ Jᵀ  (symmetric 2×2; off-diagonal uses row-0 × col-1 of Jᵀ)
    float c2d00 = jc00 * j00 + jc02 * j02;
    float c2d01 = jc01 * j11 + jc02 * j12;
    float c2d11 = jc11 * j11 + jc12 * j12;

    // add_blur: optionally compute Mip-Splatting compensation before blurring.
    // compensation = sqrt(det(Σ2d_raw) / det(Σ2d_blurred)), matches gsplat's add_blur.
    const float det_orig = calc_compensations ? (c2d00 * c2d11 - c2d01 * c2d01) : 0.0f;
    c2d00 += eps2d;
    c2d11 += eps2d;
    const float det = c2d00 * c2d11 - c2d01 * c2d01;

    if (det <= 0.0f)
        return false;

    out_compensation = calc_compensations ? sqrtf(fmaxf(0.0f, det_orig / det)) : 1.0f;

    // --- 5. Opacity threshold (uses raw opacity_cond, not compensation-scaled, matching gsplat) ---
    if (opacity_cond < ALPHA_THRESHOLD)
        return false;

    // --- 6. Opacity-aware bounding-box tightening ---
    // extend ∈ (0, 3.33]: smaller extent when the Gaussian is barely opaque
    const float extend = fminf(3.33f, sqrtf(2.0f * __logf(opacity_cond / ALPHA_THRESHOLD)));

    // --- 7. Integer pixel radii ---
    out_radius_x = (int32_t)(ceilf(extend * sqrtf(c2d00)));
    out_radius_y = (int32_t)(ceilf(extend * sqrtf(c2d11)));

    // --- 8. Project centre to pixels ---
    out_mean2d_x = fx * mc_x * rz + cx;
    out_mean2d_y = fy * mc_y * rz + cy;

    // --- 9. Frustum cull ---
    const float w = (float)image_width;
    const float h = (float)image_height;
    if (out_mean2d_x + (float)out_radius_x <= 0.0f || out_mean2d_x - (float)out_radius_x >= w ||
        out_mean2d_y + (float)out_radius_y <= 0.0f || out_mean2d_y - (float)out_radius_y >= h)
        return false;

    // --- 10. Conic: upper-tri inverse of covar2d ---
    // covar2d_inv = (1/det) * [[c2d11, -c2d01], [-c2d01, c2d00]]
    const float inv_det = 1.0f / det;
    out_conic_a = c2d11 * inv_det;  // [0,0]
    out_conic_b = -c2d01 * inv_det; // [0,1] = [1,0]
    out_conic_c = c2d00 * inv_det;  // [1,1]

    return true;
}

/**
 * covar_c = R_w2c · cov_3d_cond · R_w2cᵀ on upper-tri storage.
 */
__device__ __forceinline__ void device_covarW2C(
    float r00,
    float r01,
    float r02,
    float r10,
    float r11,
    float r12,
    float r20,
    float r21,
    float r22,
    const float s[6],
    float& out_cc00,
    float& out_cc01,
    float& out_cc02,
    float& out_cc11,
    float& out_cc12,
    float& out_cc22
)
{
    // Step 1: M = R @ cov_3d_cond  (symmetric, access via upper-tri layout)
    const float m00 = r00 * s[0] + r01 * s[1] + r02 * s[2];
    const float m01 = r00 * s[1] + r01 * s[3] + r02 * s[4];
    const float m02 = r00 * s[2] + r01 * s[4] + r02 * s[5];
    const float m10 = r10 * s[0] + r11 * s[1] + r12 * s[2];
    const float m11 = r10 * s[1] + r11 * s[3] + r12 * s[4];
    const float m12 = r10 * s[2] + r11 * s[4] + r12 * s[5];
    const float m20 = r20 * s[0] + r21 * s[1] + r22 * s[2];
    const float m21 = r20 * s[1] + r21 * s[3] + r22 * s[4];
    const float m22 = r20 * s[2] + r21 * s[4] + r22 * s[5];

    // Step 2: covar_c = M @ R^T  (only upper-tri needed; symmetry: cc10=cc01, etc.)
    out_cc00 = m00 * r00 + m01 * r01 + m02 * r02;
    out_cc01 = m00 * r10 + m01 * r11 + m02 * r12;
    out_cc02 = m00 * r20 + m01 * r21 + m02 * r22;
    out_cc11 = m10 * r10 + m11 * r11 + m12 * r12;
    out_cc12 = m10 * r20 + m11 * r21 + m12 * r22;
    out_cc22 = m20 * r20 + m21 * r21 + m22 * r22;
}

/**
 * Backward through device_covarW2C on the upper-tri parameterization: each off-diagonal is one variable
 * standing for both symmetric entries, on the camera side (upstream) and the world side (output).
 * Hence off-diagonals are halved into the symmetric matrix G and doubled out of R^T G R.
 *
 * @param r{ij}      Row i, column j of world-to-camera rotation.
 * @param d_cc{ij}   Upstream gradient of camera-space covariance (upper-tri).
 * @param d_s3d      Output: gradient w.r.t. cov_3d_cond upper-tri [6] (written, not accumulated).
 */
__device__ __forceinline__ void device_covarW2C_bwd(
    float r00,
    float r01,
    float r02,
    float r10,
    float r11,
    float r12,
    float r20,
    float r21,
    float r22,
    float d_cc00,
    float d_cc01,
    float d_cc02,
    float d_cc11,
    float d_cc12,
    float d_cc22,
    float d_s3d[6]
)
{
    const float g01 = 0.5f * d_cc01;
    const float g02 = 0.5f * d_cc02;
    const float g12 = 0.5f * d_cc12;
    // M = G @ R  (3×3)
    const float dm00 = d_cc00 * r00 + g01 * r10 + g02 * r20;
    const float dm01 = d_cc00 * r01 + g01 * r11 + g02 * r21;
    const float dm02 = d_cc00 * r02 + g01 * r12 + g02 * r22;
    const float dm10 = g01 * r00 + d_cc11 * r10 + g12 * r20;
    const float dm11 = g01 * r01 + d_cc11 * r11 + g12 * r21;
    const float dm12 = g01 * r02 + d_cc11 * r12 + g12 * r22;
    const float dm20 = g02 * r00 + g12 * r10 + d_cc22 * r20;
    const float dm21 = g02 * r01 + g12 * r11 + d_cc22 * r21;
    const float dm22 = g02 * r02 + g12 * r12 + d_cc22 * r22;
    // R^T @ M, off-diagonals doubled
    d_s3d[0] = r00 * dm00 + r10 * dm10 + r20 * dm20;
    d_s3d[1] = 2.0f * (r00 * dm01 + r10 * dm11 + r20 * dm21);
    d_s3d[2] = 2.0f * (r00 * dm02 + r10 * dm12 + r20 * dm22);
    d_s3d[3] = r01 * dm01 + r11 * dm11 + r21 * dm21;
    d_s3d[4] = 2.0f * (r01 * dm02 + r11 * dm12 + r21 * dm22);
    d_s3d[5] = r02 * dm02 + r12 * dm12 + r22 * dm22;
}

/**
 * Backward of device_persp_proj_ewa: recompute and cull identically, then += gradients to the
 * camera-space mean and covariance. With calc_compensations, up_dopac is the gradient of the compensated
 * opacity and `compensation` is written for the caller. Returns false (outputs untouched) when culled.
 */
__device__ __forceinline__ bool device_persp_proj_ewa_bwd(
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
    // ----- Replicate forward culling + Jacobian computation -----
    if (mc_z < near_plane || mc_z > far_plane)
        return false;

    const float rz = 1.0f / mc_z;
    const float rz2 = rz * rz;

    const float tan_fovx = 0.5f * (float)image_width / fx;
    const float tan_fovy = 0.5f * (float)image_height / fy;
    const float lim_x_pos = ((float)image_width - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = ((float)image_height - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    const float nx = mc_x * rz;
    const float ny = mc_y * rz;
    const float tx_clamped = mc_z * fminf(lim_x_pos, fmaxf(-lim_x_neg, nx));
    const float ty_clamped = mc_z * fminf(lim_y_pos, fmaxf(-lim_y_neg, ny));

    const float j00 = fx * rz;
    const float j02 = -fx * tx_clamped * rz2;
    const float j11 = fy * rz;
    const float j12 = -fy * ty_clamped * rz2;

    const float jc00 = j00 * cc00 + j02 * cc02;
    const float jc01 = j00 * cc01 + j02 * cc12;
    const float jc02 = j00 * cc02 + j02 * cc22;
    const float jc11 = j11 * cc11 + j12 * cc12;
    const float jc12 = j11 * cc12 + j12 * cc22;

    const float c2d00_raw = jc00 * j00 + jc02 * j02;
    const float c2d11_raw = jc11 * j11 + jc12 * j12;
    float c2d00 = c2d00_raw + eps2d;
    float c2d01 = jc01 * j11 + jc02 * j12;
    float c2d11 = c2d11_raw + eps2d;

    const float det = c2d00 * c2d11 - c2d01 * c2d01;
    if (det <= 0.0f)
        return false;

    if (opacity_cond < ALPHA_THRESHOLD)
        return false;

    const float extend = fminf(3.33f, sqrtf(2.0f * __logf(opacity_cond / ALPHA_THRESHOLD)));
    const int32_t radius_x = (int32_t)(ceilf(extend * sqrtf(c2d00)));
    const int32_t radius_y = (int32_t)(ceilf(extend * sqrtf(c2d11)));
    const float mean2d_x = fx * mc_x * rz + cx;
    const float mean2d_y = fy * mc_y * rz + cy;
    if (mean2d_x + (float)radius_x <= 0.0f || mean2d_x - (float)radius_x >= (float)image_width ||
        mean2d_y + (float)radius_y <= 0.0f || mean2d_y - (float)radius_y >= (float)image_height)
        return false;

    // ----- Backward through conic = (1/det) * [[c2d11, -c2d01], [-c2d01, c2d00]] -----
    const float inv_det = 1.0f / det;
    const float inv_det2 = inv_det * inv_det;

    float d_c2d00 = up_dc * inv_det;
    float d_c2d01 = -up_db * inv_det;
    float d_c2d11 = up_da * inv_det;

    const float d_det = -(up_da * c2d11 + up_db * (-c2d01) + up_dc * c2d00) * inv_det2;
    d_c2d00 += d_det * c2d11;
    d_c2d01 += d_det * (-2.0f * c2d01);
    d_c2d11 += d_det * c2d00;

    // ----- Backward through compensation = sqrt(det_orig / det), opac_out = opacity_cond * compensation -----
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

    // ----- Backward through mean2d = (fx*mc_x/mc_z + cx, fy*mc_y/mc_z + cy) -----
    d_mc_x += up_dm2d_x * fx * rz;
    d_mc_y += up_dm2d_y * fy * rz;
    d_mc_z += up_dm2d_x * (-fx * mc_x * rz2) + up_dm2d_y * (-fy * mc_y * rz2);

    // ----- Backward through c2d = JC * J^T -----
    const float d_jc00 = d_c2d00 * j00;
    const float d_jc02 = d_c2d00 * j02 + d_c2d01 * j12;
    const float d_jc01 = d_c2d01 * j11;
    const float d_jc11 = d_c2d11 * j11;
    const float d_jc12 = d_c2d11 * j12;

    float d_j00 = d_c2d00 * jc00;
    float d_j02 = d_c2d00 * jc02;
    float d_j11 = d_c2d01 * jc01 + d_c2d11 * jc11;
    float d_j12 = d_c2d01 * jc02 + d_c2d11 * jc12;

    // ----- Backward through JC = J @ Sigma_c -----
    d_cc00 += d_jc00 * j00;
    d_cc01 += d_jc01 * j00;
    d_cc02 += d_jc00 * j02 + d_jc02 * j00;
    d_cc11 += d_jc11 * j11;
    d_cc12 += d_jc01 * j02 + d_jc11 * j12 + d_jc12 * j11;
    d_cc22 += d_jc02 * j02 + d_jc12 * j12;

    d_j00 += d_jc00 * cc00 + d_jc01 * cc01 + d_jc02 * cc02;
    d_j02 += d_jc00 * cc02 + d_jc01 * cc12 + d_jc02 * cc22;
    d_j11 += d_jc11 * cc11 + d_jc12 * cc12;
    d_j12 += d_jc11 * cc12 + d_jc12 * cc22;

    // ----- Backward through J elements w.r.t. mc_z -----
    // j00 = fx/mc_z:  d_mc_z += -d_j00 * fx/mc_z^2 = -d_j00 * j00 * rz
    d_mc_z += -d_j00 * j00 * rz;
    d_mc_z += -d_j11 * j11 * rz;

    // j02 = -fx * tx_clamped / mc_z^2  →  d_tx_clamped, d_rz2 contributions
    const float d_tx_clamped = d_j02 * (-fx * rz2);
    const float d_ty_clamped = d_j12 * (-fy * rz2);
    const float d_rz2 = d_j02 * (-fx * tx_clamped) + d_j12 * (-fy * ty_clamped);
    d_mc_z += d_rz2 * (-2.0f * rz2 * rz); // rz2 = mc_z^{-2}: d_rz2/d_mc_z = -2/mc_z^3

    // ----- Backward through clamp: tx_clamped = mc_z * clamp(mc_x/mc_z, ...) -----
    if (nx > lim_x_pos)
        d_mc_z += d_tx_clamped * lim_x_pos;
    else if (nx < -lim_x_neg)
        d_mc_z += d_tx_clamped * (-lim_x_neg);
    else
        d_mc_x += d_tx_clamped;

    if (ny > lim_y_pos)
        d_mc_z += d_ty_clamped * lim_y_pos;
    else if (ny < -lim_y_neg)
        d_mc_z += d_ty_clamped * (-lim_y_neg);
    else
        d_mc_y += d_ty_clamped;

    return true;
}
