/**
 * Fused 4D conditioning + SH + EWA projection backward, one thread per (Gaussian, camera) pair
 * (gauss_idx = idx % N, cam_idx = idx / N). Recomputes the forward and runs device_persp_proj_ewa_bwd,
 * device_covarW2C_bwd, the appearance leaf backward and the conditioning backward; parameter
 * gradients are atomicAdd-ed across cameras.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

#include "conditioning.cuh"
#include "ewa_4d.cuh"
#include "ewa_4d_lens.cuh"
#include "ewa_4d_lens_limits.cuh"
#include "sh4d.cuh"

using at::Tensor;

template <ProjectionFamily F, bool kLens, AppearanceKind M>
__launch_bounds__(256, 2) __global__ void ewa_4d_fused_conditioned_projection_bwd_kernel(
    // --- Upstream gradients [C, N, ...] ---
    const float* __restrict__ d_means2d, // [C, N, 2]
    const float* __restrict__ d_conics,  // [C, N, 3]
    const float* __restrict__ d_colors,  // [C, N, 3]
    const float* __restrict__ d_opac,    // [C, N]
    // --- Forward inputs (for recomputation) ---
    const float* __restrict__ quats_l,       // [N, 4] wxyz, unnormalized
    const float* __restrict__ quats_r,       // [N, 4] wxyz, unnormalized
    const float* __restrict__ scales_4d_log, // [N, 4] log-space
    const float* __restrict__ means,         // [N, 3]
    const float* __restrict__ mu_t,          // [N]
    const float* __restrict__ opacities,     // [N] raw logit
    const float* __restrict__ sh0,           // [N, 1, 3]
    const float* __restrict__ shN,           // [N, K_max - 1, 3]
    const float* __restrict__ sh_temporal,   // [N, T, 3] or NULL
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    const float* __restrict__ viewmats,   // [C, 4, 4] row-major
    const float* __restrict__ Ks,         // [C, 3, 3] row-major
    const float* __restrict__ dist,       // [C, n_slots] lens distortion; NULL when !kLens
    const float* __restrict__ clamp_lims, // [C, 4] Brown–Conrady clamp limits; NULL otherwise
    int C,
    float tau,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    int image_width,
    int image_height,
    bool calc_compensations,
    // --- Output gradients [N, ...] (atomicAdd across cameras) ---
    float* __restrict__ d_quats_l,       // [N, 4]
    float* __restrict__ d_quats_r,       // [N, 4]
    float* __restrict__ d_scales_4d_log, // [N, 4]
    float* __restrict__ d_means_out,     // [N, 3]
    float* __restrict__ d_mu_t,          // [N]
    float* __restrict__ d_opacities_out, // [N]
    float* __restrict__ d_sh0_out,       // [N, 1, 3]
    float* __restrict__ d_shN_out,       // [N, K_max - 1, 3]
    float* __restrict__ d_sh_temporal,   // [N, T, 3] or NULL
    int N
)
{
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= C * N)
        return;

    const int gauss_idx = global_idx % N;
    const int cam_idx = global_idx / N;
    const int out_idx = cam_idx * N + gauss_idx;

    // ====================================================================
    // RECOMPUTE FORWARD — Stage 1: Temporal Conditioning
    // Retain all intermediates needed by the Stage 1 backward.
    // ====================================================================

    float ql_raw[4], qr_raw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        ql_raw[i] = quats_l[gauss_idx * 4 + i];
        qr_raw[i] = quats_r[gauss_idx * 4 + i];
    }
    float ql[4], qr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        ql[i] = ql_raw[i];
        qr[i] = qr_raw[i];
    }
    const float ql_norm = quat_normalize(ql);
    const float qr_norm = quat_normalize(qr);

    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        s[i] = expf(scales_4d_log[gauss_idx * 4 + i]);

    float A[4][4];
    build_A_rot4d(ql, qr, A);
    float cov_3d_marg[6], cov_xt[3], cov_t_marg;
    build_sigma_schur(A, s, cov_3d_marg, cov_xt, cov_t_marg);

    const float mu_t_val = mu_t[gauss_idx];
    const float m0 = means[gauss_idx * 3 + 0];
    const float m1 = means[gauss_idx * 3 + 1];
    const float m2 = means[gauss_idx * 3 + 2];
    const float dt = tau - mu_t_val;
    const float cov_t_marg_safe = cov_t_marg + (float)TEMPORAL_VARIANCE_EPS;
    const float inv_cov_t_marg = 1.0f / cov_t_marg_safe;
    const float mean_corr_scale = dt * inv_cov_t_marg;
    const float mx = m0 + cov_xt[0] * mean_corr_scale;
    const float my = m1 + cov_xt[1] * mean_corr_scale;
    const float mz = m2 + cov_xt[2] * mean_corr_scale;

    // Σ as the forward draws it (from L), so the projection Jacobian sees the same covariance.
    float cov_L[6], cov_3d_cond[6];
    device_conditioned_factor(A, s, cov_xt, cov_t_marg, cov_eps, cov_L);
    device_gram_lower(cov_L, cov_3d_cond);

    const float wt = expf(-0.5f * dt * dt * inv_cov_t_marg);
    const float opacity_base = sigmoid(opacities[gauss_idx]);
    const bool passes = (prefilter_threshold <= 0.0f) || (wt >= prefilter_threshold);
    const float opacity_cond = passes ? opacity_base * wt : 0.0f;

    // RECOMPUTE FORWARD — Stages 2 & 3: Camera extraction and W2C transform

    const float* vm = viewmats + cam_idx * 16;
    const float tx = vm[3], ty = vm[7], tz = vm[11];
    const float r00 = vm[0], r01 = vm[1], r02 = vm[2];
    const float r10 = vm[4], r11 = vm[5], r12 = vm[6];
    const float r20 = vm[8], r21 = vm[9], r22 = vm[10];

    // Camera position: campos = -R^T @ t
    const float cam_x = -(r00 * tx + r10 * ty + r20 * tz);
    const float cam_y = -(r01 * tx + r11 * ty + r21 * tz);
    const float cam_z = -(r02 * tx + r12 * ty + r22 * tz);

    // View direction for SH evaluation (unnormalized)
    const float dir_x = mx - cam_x;
    const float dir_y = my - cam_y;
    const float dir_z = mz - cam_z;
    const float dir_t = dt; // same as tau - mu_t_val

    // Camera-space mean and covariance
    const float mc_x = r00 * mx + r01 * my + r02 * mz + tx;
    const float mc_y = r10 * mx + r11 * my + r12 * mz + ty;
    const float mc_z = r20 * mx + r21 * my + r22 * mz + tz;
    float cc00, cc01, cc02, cc11, cc12, cc22;
    device_covarW2C(r00, r01, r02, r10, r11, r12, r20, r21, r22, cov_3d_cond, cc00, cc01, cc02, cc11, cc12, cc22);

    // ====================================================================
    // BACKWARD — Stage 4: EWA Projection (device_persp_proj_ewa_bwd)
    // Returns false if culled — no gradients to accumulate.
    // ====================================================================

    const float* K_ptr = Ks + cam_idx * 9;
    float d_mc_x = 0.0f, d_mc_y = 0.0f, d_mc_z = 0.0f;
    float d_cc00 = 0.0f, d_cc01 = 0.0f, d_cc02 = 0.0f;
    float d_cc11 = 0.0f, d_cc12 = 0.0f, d_cc22 = 0.0f;
    float compensation = 1.0f;

    const bool visible = device_ewa_project_bwd<F, kLens>(
        mc_x,
        mc_y,
        mc_z,
        cc00,
        cc01,
        cc02,
        cc11,
        cc12,
        cc22,
        K_ptr[0],
        K_ptr[4],
        K_ptr[2],
        K_ptr[5], // fx, fy, cx, cy
        device_ewa_lens<F, kLens>(dist, clamp_lims, cam_idx),
        eps2d,
        opacity_cond,
        near_plane,
        far_plane,
        image_width,
        image_height,
        d_means2d[out_idx * 2 + 0],
        d_means2d[out_idx * 2 + 1],
        d_conics[out_idx * 3 + 0],
        d_conics[out_idx * 3 + 1],
        d_conics[out_idx * 3 + 2],
        calc_compensations,
        d_opac[out_idx],
        compensation,
        d_mc_x,
        d_mc_y,
        d_mc_z,
        d_cc00,
        d_cc01,
        d_cc02,
        d_cc11,
        d_cc12,
        d_cc22
    );
    if (!visible)
        return;

    // ====================================================================
    // BACKWARD — Opacity: opacity_cond = opacity_base * wt  (when passes)
    // In antialiased mode: opac_out = opacity_cond * compensation, so upstream
    // gradient w.r.t. opacity_cond is d_opac * compensation.
    // ====================================================================

    float d_opac_local = 0.0f, d_wt_local = 0.0f;
    if (passes) {
        const float up_dopac = d_opac[out_idx] * compensation;
        d_opac_local = up_dopac * wt * sigmoid_grad(opacity_base);
        d_wt_local = up_dopac * opacity_base;
    }

    // BACKWARD — Stage 3: World-to-Camera Transform

    // Covariance VJP: d_cov_3d_cond = R^T @ D @ R
    float d_s3d[6];
    device_covarW2C_bwd(
        r00,
        r01,
        r02,
        r10,
        r11,
        r12,
        r20,
        r21,
        r22,
        d_cc00,
        d_cc01,
        d_cc02,
        d_cc11,
        d_cc12,
        d_cc22,
        d_s3d
    );

    // Mean VJP: d_mean_world = R^T @ d_mc
    float d_mx = r00 * d_mc_x + r10 * d_mc_y + r20 * d_mc_z;
    float d_my = r01 * d_mc_x + r11 * d_mc_y + r21 * d_mc_z;
    float d_mz = r02 * d_mc_x + r12 * d_mc_y + r22 * d_mc_z;

    // ====================================================================
    // BACKWARD — Stage 2: SH Color Evaluation (AppearanceOps<M>::eval_bwd)
    // Accumulates SH coefficient gradients via atomicAdd.
    // Adds unnormalized direction gradient to d_mx/y/z (dir = mean_world - cam_pos).
    // ====================================================================
    const ShCoeffs sh{sh0 + gauss_idx * 3, shN + gauss_idx * shN_stride};
    const ShGrad d_sh{d_sh0_out + gauss_idx * 3, d_shN_out + gauss_idx * shN_stride};
    const float* sh_temp_ptr =
        (sh_temporal != nullptr && n_temporal_terms > 0) ? sh_temporal + gauss_idx * temporal_stride : nullptr;
    float* d_sh_temp_ptr =
        (d_sh_temporal != nullptr && n_temporal_terms > 0) ? d_sh_temporal + gauss_idx * temporal_stride : nullptr;

    AppearanceOps<M>::eval_bwd(
        sh_degree,
        n_temporal_terms,
        temporal_stride,
        dir_x,
        dir_y,
        dir_z,
        dir_t,
        sh,
        sh_temp_ptr,
        d_colors[out_idx * 3 + 0],
        d_colors[out_idx * 3 + 1],
        d_colors[out_idx * 3 + 2],
        d_sh,
        d_sh_temp_ptr,
        d_mx,
        d_my,
        d_mz // dir = mean_world - cam_pos → d_dir flows into d_mean_world
    );

    // BACKWARD — Stage 1: conditioning. Mean and gate through the Schur blocks; the covariance through N.
    const float d_cov_3d_marg[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // covariance: factored backward below
    float d_cov_xt[3] = {0.f, 0.f, 0.f};
    // d_cov_xt from mean: mx = m0 + cov_xt[0]*mean_corr_scale,  mean_corr_scale = dt/cov_t_marg_safe
    d_cov_xt[0] += d_mx * dt * inv_cov_t_marg;
    d_cov_xt[1] += d_my * dt * inv_cov_t_marg;
    d_cov_xt[2] += d_mz * dt * inv_cov_t_marg;
    float d_cov_t_marg_safe = 0.f;

    // d_cov_t_marg_safe from mean: mean_corr_scale = dt/cov_t_marg_safe
    d_cov_t_marg_safe -=
        (d_mx * cov_xt[0] + d_my * cov_xt[1] + d_mz * cov_xt[2]) * dt * inv_cov_t_marg * inv_cov_t_marg;

    // d_cov_t_marg_safe from wt: wt = exp(-0.5*dt^2/cov_t_marg_safe)
    const float d_wt_times_wt = d_wt_local * wt;
    d_cov_t_marg_safe += d_wt_times_wt * 0.5f * dt * dt * inv_cov_t_marg * inv_cov_t_marg;

    // d_dt from mean and wt
    const float d_dt = (d_mx * cov_xt[0] + d_my * cov_xt[1] + d_mz * cov_xt[2]) * inv_cov_t_marg +
                       d_wt_times_wt * (-dt * inv_cov_t_marg);

    // cov_t_marg_safe = cov_t_marg + TEMPORAL_VARIANCE_EPS  →  d_cov_t_marg = d_cov_t_marg_safe
    const float d_cov_t_marg = d_cov_t_marg_safe;

    // dt = tau - mu_t_val  →  d_mu_t_local = -d_dt

    const float d_mu_t_local = -d_dt;

    // BACKWARD — build_sigma_schur, build_A_rot4d, quat_normalize  (conditioning.cuh)

    float d_scales_4d_log_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d_A[4][4];
    build_sigma_schur_bwd(A, s, d_cov_3d_marg, d_cov_xt, d_cov_t_marg, d_A, d_scales_4d_log_local);
    device_conditioned_cov_bwd(A, s, cov_xt, cov_t_marg, d_s3d, d_A, d_scales_4d_log_local);

    float d_ql_hat[4], d_qr_hat[4];
    build_A_rot4d_bwd(ql, qr, d_A, d_ql_hat, d_qr_hat);

    float d_ql_raw[4], d_qr_raw[4];
    quat_normalize_bwd(ql, d_ql_hat, ql_norm, d_ql_raw);
    quat_normalize_bwd(qr, d_qr_hat, qr_norm, d_qr_raw);

    // ACCUMULATE via atomicAdd (reduction across cameras)

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        atomicAdd(&d_quats_l[gauss_idx * 4 + i], d_ql_raw[i]);
        atomicAdd(&d_quats_r[gauss_idx * 4 + i], d_qr_raw[i]);
        atomicAdd(&d_scales_4d_log[gauss_idx * 4 + i], d_scales_4d_log_local[i]);
    }

    atomicAdd(&d_means_out[gauss_idx * 3 + 0], d_mx);
    atomicAdd(&d_means_out[gauss_idx * 3 + 1], d_my);
    atomicAdd(&d_means_out[gauss_idx * 3 + 2], d_mz);

    atomicAdd(&d_mu_t[gauss_idx], d_mu_t_local);
    atomicAdd(&d_opacities_out[gauss_idx], d_opac_local);
}

/**
 * Launch ewa_4d_fused_conditioned_projection_bwd_kernel and return gradient tensors.
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
launch_ewa_4d_fused_conditioned_projection_bwd_kernel(
    const Tensor& d_means2d, // [C, N, 2]
    const Tensor& d_conics,  // [C, N, 3]
    const Tensor& d_colors,  // [C, N, 3]
    const Tensor& d_opac,    // [C, N]
    const Tensor& quats_l,
    const Tensor& quats_r,
    const Tensor& scales_4d_log,
    const Tensor& means,
    const Tensor& mu_t,
    const Tensor& opacities,
    const Tensor& sh0,
    const Tensor& shN,
    const Tensor& sh_temporal,
    int sh_degree,
    int appearance_kind,
    int n_temporal_terms,
    int temporal_stride,
    const Tensor& viewmats,
    const Tensor& Ks,
    const Tensor& dist,
    int family,
    float tau,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    int image_width,
    int image_height,
    bool calc_compensations
)
{
    const int N = (int)means.size(0);
    const int C = (int)viewmats.size(0);
    const EwaLensArgs lens_args = ewa_lens_args(Ks, dist, family, image_width, image_height);
    const int total = C * N;

    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");

    auto opts_f = means.options();

    // Zero-initialized gradient tensors
    Tensor d_quats_l_out = at::zeros({N, 4}, opts_f);
    Tensor d_quats_r_out = at::zeros({N, 4}, opts_f);
    Tensor d_scales_4d_log_out = at::zeros({N, 4}, opts_f);
    Tensor d_means_out = at::zeros({N, 3}, opts_f);
    Tensor d_mu_t_out = at::zeros({N}, opts_f);
    Tensor d_opacities_out = at::zeros({N}, opts_f);
    Tensor d_sh0_out = at::zeros(sh0.sizes(), sh0.options()); // contiguous: the kernel indexes it densely
    Tensor d_shN_out = at::zeros(shN.sizes(), shN.options());
    Tensor d_sh_temporal_out = (sh_temporal.numel() > 0) ? at::zeros_like(sh_temporal) : at::zeros({0}, opts_f);

    if (total == 0) {
        return {
            d_quats_l_out,
            d_quats_r_out,
            d_scales_4d_log_out,
            d_means_out,
            d_mu_t_out,
            d_opacities_out,
            d_sh0_out,
            d_shN_out,
            d_sh_temporal_out
        };
    }

    const float* sh_temporal_ptr = (sh_temporal.numel() > 0) ? sh_temporal.contiguous().data_ptr<float>() : nullptr;
    float* d_sh_temporal_ptr = (d_sh_temporal_out.numel() > 0) ? d_sh_temporal_out.data_ptr<float>() : nullptr;

    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();

    ANYCAM4D_DISPATCH_EWA_LENS(
        lens_args.has_lens,
        family,
        ANYCAM4D_DISPATCH_APPEARANCE(
            appearance_kind,
            ewa_4d_fused_conditioned_projection_bwd_kernel<F, kLens, M><<<grid_size, block_size, 0, stream>>>(
                d_means2d.contiguous().data_ptr<float>(),
                d_conics.contiguous().data_ptr<float>(),
                d_colors.contiguous().data_ptr<float>(),
                d_opac.contiguous().data_ptr<float>(),
                quats_l.contiguous().data_ptr<float>(),
                quats_r.contiguous().data_ptr<float>(),
                scales_4d_log.contiguous().data_ptr<float>(),
                means.contiguous().data_ptr<float>(),
                mu_t.contiguous().data_ptr<float>(),
                opacities.contiguous().data_ptr<float>(),
                sh0.contiguous().data_ptr<float>(),
                shN.contiguous().data_ptr<float>(),
                sh_temporal_ptr,
                sh_degree,
                (int)shN.size(1) * 3,
                n_temporal_terms,
                temporal_stride,
                viewmats.contiguous().data_ptr<float>(),
                Ks.contiguous().data_ptr<float>(),
                lens_args.dist_ptr,
                lens_args.lims_ptr,
                C,
                tau,
                cov_eps,
                eps2d,
                near_plane,
                far_plane,
                prefilter_threshold,
                image_width,
                image_height,
                calc_compensations,
                d_quats_l_out.data_ptr<float>(),
                d_quats_r_out.data_ptr<float>(),
                d_scales_4d_log_out.data_ptr<float>(),
                d_means_out.data_ptr<float>(),
                d_mu_t_out.data_ptr<float>(),
                d_opacities_out.data_ptr<float>(),
                d_sh0_out.data_ptr<float>(),
                d_shN_out.data_ptr<float>(),
                d_sh_temporal_ptr,
                N
            )
        )
    );

    return {
        d_quats_l_out,
        d_quats_r_out,
        d_scales_4d_log_out,
        d_means_out,
        d_mu_t_out,
        d_opacities_out,
        d_sh0_out,
        d_shN_out,
        d_sh_temporal_out
    };
}
