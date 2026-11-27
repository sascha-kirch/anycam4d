/**
 * Fused 4D conditioning + 4D SH backward through the differentiable outputs only; the 2D projection
 * outputs are stop-gradient. Per-Gaussian upstreams (d_means_3d_cond, d_cov_factors, d_opacities) are applied by
 * the cam_idx == 0 thread, per-camera d_colors by every thread; linearity makes the atomicAdd-ed sums exact.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

#include "conditioning.cuh"
#include "sh4d.cuh"

using at::Tensor;

template <AppearanceKind M>
__launch_bounds__(256, 2) __global__ void ut_4d_fused_projection_bwd_kernel(
    // Upstream gradients
    const float* __restrict__ d_means3d_up,     // [N, 3]
    const float* __restrict__ d_cov_factors_up, // [N, 6]
    const float* __restrict__ d_opacities_up,   // [N]
    const float* __restrict__ d_colors_up,      // [C, N, 3]
    // Forward inputs (for recomputation)
    const float* __restrict__ quats_l,
    const float* __restrict__ quats_r,
    const float* __restrict__ scales_4d_log,
    const float* __restrict__ means,
    const float* __restrict__ mu_t,
    const float* __restrict__ opacities_in,
    const float* __restrict__ sh0, // [N, 1, 3]
    const float* __restrict__ shN, // [N, K_max - 1, 3]
    const float* __restrict__ sh_temporal,
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    const float* __restrict__ viewmats,
    int C,
    float tau,
    float cov_eps,
    float prefilter_threshold,
    // Output gradients (atomicAdd)
    float* __restrict__ d_quats_l_out,
    float* __restrict__ d_quats_r_out,
    float* __restrict__ d_scales_4d_log_out,
    float* __restrict__ d_means_out,
    float* __restrict__ d_mu_t_out,
    float* __restrict__ d_opacities_out,
    float* __restrict__ d_sh0_out, // [N, 1, 3]
    float* __restrict__ d_shN_out, // [N, K_max - 1, 3]
    float* __restrict__ d_sh_temporal_out,
    int N
)
{
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= C * N)
        return;

    const int gauss_idx = global_idx % N;
    const int cam_idx = global_idx / N;
    const int out_idx = cam_idx * N + gauss_idx;

    // RECOMPUTE FORWARD — Stage 1 conditioning intermediates

    float ql[4], qr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        ql[i] = quats_l[gauss_idx * 4 + i];
        qr[i] = quats_r[gauss_idx * 4 + i];
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

    const float wt = expf(-0.5f * dt * dt * inv_cov_t_marg);
    const float opacity_base = sigmoid(opacities_in[gauss_idx]);
    const bool passes = (prefilter_threshold <= 0.0f) || (wt >= prefilter_threshold);

    // ====================================================================
    // BACKWARD — Stage 3: 4D SH Color Evaluation
    // Per-camera contribution; atomicAdd into d_sh0/d_shN/d_sh_temporal.
    // Direction backward adds to local d_mx/d_my/d_mz.
    // ====================================================================

    const float* vm = viewmats + cam_idx * 16;
    const float tx_vm = vm[3], ty_vm = vm[7], tz_vm = vm[11];
    const float r00 = vm[0], r01 = vm[1], r02 = vm[2];
    const float r10 = vm[4], r11 = vm[5], r12 = vm[6];
    const float r20 = vm[8], r21 = vm[9], r22 = vm[10];
    const float cam_x = -(r00 * tx_vm + r10 * ty_vm + r20 * tz_vm);
    const float cam_y = -(r01 * tx_vm + r11 * ty_vm + r21 * tz_vm);
    const float cam_z = -(r02 * tx_vm + r12 * ty_vm + r22 * tz_vm);

    const float dir_x = mx - cam_x;
    const float dir_y = my - cam_y;
    const float dir_z = mz - cam_z;
    const float dir_t = dt;

    float d_mx = 0.f, d_my = 0.f, d_mz = 0.f;
    const ShCoeffs sh{sh0 + gauss_idx * 3, shN + gauss_idx * shN_stride};
    const ShGrad d_sh{d_sh0_out + gauss_idx * 3, d_shN_out + gauss_idx * shN_stride};
    const float* sh_temp_ptr =
        (sh_temporal != nullptr && n_temporal_terms > 0) ? sh_temporal + gauss_idx * temporal_stride : nullptr;
    float* d_sh_temp_ptr = (d_sh_temporal_out != nullptr && n_temporal_terms > 0)
                               ? d_sh_temporal_out + gauss_idx * temporal_stride
                               : nullptr;

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
        d_colors_up[out_idx * 3 + 0],
        d_colors_up[out_idx * 3 + 1],
        d_colors_up[out_idx * 3 + 2],
        d_sh,
        d_sh_temp_ptr,
        d_mx,
        d_my,
        d_mz
    );

    // Per-Gaussian upstreams: applied by cam_idx == 0 thread only

    float d_factor[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float d_opacity_cond = 0.f;
    if (cam_idx == 0) {
        d_mx += d_means3d_up[gauss_idx * 3 + 0];
        d_my += d_means3d_up[gauss_idx * 3 + 1];
        d_mz += d_means3d_up[gauss_idx * 3 + 2];
#pragma unroll
        for (int i = 0; i < 6; ++i)
            d_factor[i] = d_cov_factors_up[gauss_idx * 6 + i];
        d_opacity_cond = d_opacities_up[gauss_idx];
    }

    // BACKWARD — Opacity: opacity_cond = passes ? opacity_base * wt : 0

    float d_opac_local = 0.f, d_wt_local = 0.f;
    if (passes) {
        d_opac_local = d_opacity_cond * wt * sigmoid_grad(opacity_base);
        d_wt_local = d_opacity_cond * opacity_base;
    }

    // ====================================================================
    // BACKWARD — Stage 1: Schur Conditioning
    // Mean and gate through the Schur blocks; the covariance factor through N (device_conditioned_factor_bwd).
    // ====================================================================
    const float d_cov_3d_marg[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // covariance: factored backward below
    float d_cov_xt[3] = {0.f, 0.f, 0.f};
    d_cov_xt[0] += d_mx * dt * inv_cov_t_marg;
    d_cov_xt[1] += d_my * dt * inv_cov_t_marg;
    d_cov_xt[2] += d_mz * dt * inv_cov_t_marg;
    float d_cov_t_marg_safe = 0.f;
    d_cov_t_marg_safe -=
        (d_mx * cov_xt[0] + d_my * cov_xt[1] + d_mz * cov_xt[2]) * dt * inv_cov_t_marg * inv_cov_t_marg;

    const float d_wt_times_wt = d_wt_local * wt;
    d_cov_t_marg_safe += d_wt_times_wt * 0.5f * dt * dt * inv_cov_t_marg * inv_cov_t_marg;

    const float d_dt = (d_mx * cov_xt[0] + d_my * cov_xt[1] + d_mz * cov_xt[2]) * inv_cov_t_marg +
                       d_wt_times_wt * (-dt * inv_cov_t_marg);

    const float d_cov_t_marg = d_cov_t_marg_safe;
    const float d_mu_t_local = -d_dt;

    float d_scales_4d_log_local[4] = {0.f, 0.f, 0.f, 0.f};
    float d_A[4][4];
    build_sigma_schur_bwd(A, s, d_cov_3d_marg, d_cov_xt, d_cov_t_marg, d_A, d_scales_4d_log_local);
    if (cam_idx == 0)
        device_conditioned_factor_bwd(A, s, cov_xt, cov_t_marg, cov_eps, d_factor, d_A, d_scales_4d_log_local);

    float d_ql_hat[4], d_qr_hat[4];
    build_A_rot4d_bwd(ql, qr, d_A, d_ql_hat, d_qr_hat);

    float d_ql_raw[4], d_qr_raw[4];
    quat_normalize_bwd(ql, d_ql_hat, ql_norm, d_ql_raw);
    quat_normalize_bwd(qr, d_qr_hat, qr_norm, d_qr_raw);

    // ACCUMULATE via atomicAdd (reduction across cameras)

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        atomicAdd(&d_quats_l_out[gauss_idx * 4 + i], d_ql_raw[i]);
        atomicAdd(&d_quats_r_out[gauss_idx * 4 + i], d_qr_raw[i]);
        atomicAdd(&d_scales_4d_log_out[gauss_idx * 4 + i], d_scales_4d_log_local[i]);
    }

    atomicAdd(&d_means_out[gauss_idx * 3 + 0], d_mx);
    atomicAdd(&d_means_out[gauss_idx * 3 + 1], d_my);
    atomicAdd(&d_means_out[gauss_idx * 3 + 2], d_mz);

    atomicAdd(&d_mu_t_out[gauss_idx], d_mu_t_local);
    atomicAdd(&d_opacities_out[gauss_idx], d_opac_local);
}

/**
 * Launch ut_4d_fused_projection_bwd_kernel and return gradient tensors.
 *
 * Returns gradients in the order matching FusedUTFunction.forward's
 * differentiable inputs:
 *   (d_quats_l, d_quats_r, d_scales_4d_log, d_means, d_mu_t, d_opacities,
 *    d_sh0, d_shN, d_sh_temporal)
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> ut_4d_fused_projection_bwd(
    const Tensor& d_means_3d_cond, // [N, 3]
    const Tensor& d_cov_factors,   // [N, 6]
    const Tensor& d_opacities,     // [N]
    const Tensor& d_colors,        // [C, N, 3]
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
    float tau,
    float cov_eps,
    float prefilter_threshold
)
{
    const int N = (int)means.size(0);
    const int C = (int)viewmats.size(0);
    const int total = C * N;

    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");

    auto opts_f = means.options();

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

    ANYCAM4D_DISPATCH_APPEARANCE(
        appearance_kind,
        ut_4d_fused_projection_bwd_kernel<M><<<grid_size, block_size, 0, stream>>>(
            d_means_3d_cond.contiguous().data_ptr<float>(),
            d_cov_factors.contiguous().data_ptr<float>(),
            d_opacities.contiguous().data_ptr<float>(),
            d_colors.contiguous().data_ptr<float>(),
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
            C,
            tau,
            cov_eps,
            prefilter_threshold,
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
