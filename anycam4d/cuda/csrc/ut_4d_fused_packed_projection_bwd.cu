/**
 * Packed fused UT backward through SH and conditioning; the 2D projection outputs are stop-gradient.
 *
 * One thread per visible pair recomputes conditioning, runs the SH backward with d_colors[cam * N + gauss],
 * then the conditioning backward with the dense per-Gaussian upstreams divided by vis_count, which is
 * exact because that backward is linear. Gradients are atomicAdd-ed into dense [N, ...] buffers.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

#include "conditioning.cuh"
#include "sh4d.cuh"

using at::Tensor;

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/**
 * Backward over nnz visible pairs; vis_count [N] is the number of visible cameras per Gaussian.
 */
template <AppearanceKind M>
__launch_bounds__(256, 2) __global__ void ut_4d_fused_packed_projection_bwd_kernel(
    // Dense upstream gradients
    const float* __restrict__ d_colors,         // [C, N, 3]
    const float* __restrict__ d_means3d_up,     // [N, 3]
    const float* __restrict__ d_cov_factors_up, // [N, 6]
    const float* __restrict__ d_opacities_up,   // [N]
    // Packed pair indices
    const int64_t* __restrict__ camera_ids,   // [nnz]
    const int64_t* __restrict__ gaussian_ids, // [nnz]
    const int64_t* __restrict__ vis_count,    // [N]
    // Forward inputs (for recomputation)
    const float* __restrict__ quats_l,       // [N, 4]
    const float* __restrict__ quats_r,       // [N, 4]
    const float* __restrict__ scales_4d_log, // [N, 4]
    const float* __restrict__ means,         // [N, 3]
    const float* __restrict__ mu_t,          // [N]
    const float* __restrict__ opacities_in,  // [N]
    const float* __restrict__ sh0,           // [N, 1, 3]
    const float* __restrict__ shN,           // [N, K_max - 1, 3]
    const float* __restrict__ sh_temporal,   // [N, T, 3] or NULL
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    const float* __restrict__ viewmats, // [C, 4, 4]
    int C,
    int N,
    float tau,
    float cov_eps,
    float prefilter_threshold,
    int nnz,
    // Output gradients [N, ...] (atomicAdd)
    float* __restrict__ d_quats_l_out,       // [N, 4]
    float* __restrict__ d_quats_r_out,       // [N, 4]
    float* __restrict__ d_scales_4d_log_out, // [N, 4]
    float* __restrict__ d_means_out,         // [N, 3]
    float* __restrict__ d_mu_t_out,          // [N]
    float* __restrict__ d_opacities_out,     // [N]
    float* __restrict__ d_sh0_out,           // [N, 1, 3]
    float* __restrict__ d_shN_out,           // [N, K_max - 1, 3]
    float* __restrict__ d_sh_temporal_out    // [N, T, 3] or NULL
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz)
        return;

    const int64_t cam_idx = camera_ids[idx];
    const int64_t gauss_idx = gaussian_ids[idx];

    // RECOMPUTE FORWARD — Stage 1: Temporal Conditioning

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
    const float opacity_cond = passes ? opacity_base * wt : 0.0f;

    // RECOMPUTE camera position and view direction (for SH backward)

    const float* vm = viewmats + cam_idx * 16;
    const float tx = vm[3], ty = vm[7], tz = vm[11];
    const float r00 = vm[0], r01 = vm[1], r02 = vm[2];
    const float r10 = vm[4], r11 = vm[5], r12 = vm[6];
    const float r20 = vm[8], r21 = vm[9], r22 = vm[10];

    const float cam_x = -(r00 * tx + r10 * ty + r20 * tz);
    const float cam_y = -(r01 * tx + r11 * ty + r21 * tz);
    const float cam_z = -(r02 * tx + r12 * ty + r22 * tz);

    const float dir_x = mx - cam_x;
    const float dir_y = my - cam_y;
    const float dir_z = mz - cam_z;
    const float dir_t = dt;

    // ====================================================================
    // BACKWARD — Stage 2: 4D SH Color Evaluation
    // Reads d_colors at dense index (cam_idx * N + gauss_idx).
    // Accumulates d_sh0 / d_shN / d_sh_temporal via atomicAdd (inside device fn).
    // Contributes to d_mx, d_my, d_mz through the direction gradient.
    // ====================================================================

    float d_mx = 0.0f, d_my = 0.0f, d_mz = 0.0f;
    const ShCoeffs sh{sh0 + gauss_idx * 3, shN + gauss_idx * shN_stride};
    const ShGrad d_sh{d_sh0_out + gauss_idx * 3, d_shN_out + gauss_idx * shN_stride};
    const float* sh_temp_ptr =
        (sh_temporal != nullptr && n_temporal_terms > 0) ? sh_temporal + gauss_idx * temporal_stride : nullptr;
    float* d_sh_temp_ptr = (d_sh_temporal_out != nullptr && n_temporal_terms > 0)
                               ? d_sh_temporal_out + gauss_idx * temporal_stride
                               : nullptr;

    const int64_t color_idx = (cam_idx * N + gauss_idx) * 3;
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
        d_colors[color_idx + 0],
        d_colors[color_idx + 1],
        d_colors[color_idx + 2],
        d_sh,
        d_sh_temp_ptr,
        d_mx,
        d_my,
        d_mz
    );

    // ====================================================================
    // Per-Gaussian upstream gradients — scaled by 1 / vis_count
    //
    // The conditioning backward is camera-independent.  Since packed pairs
    // iterate over nnz = sum(vis_count[n]) items, applying the full per-Gaussian
    // upstream vis_count[n] times would over-count by vis_count[n].
    // Dividing by vis_count before accumulation gives the correct total.
    // ====================================================================

    const float inv_vc = 1.0f / static_cast<float>(vis_count[gauss_idx]);

    d_mx += d_means3d_up[gauss_idx * 3 + 0] * inv_vc;
    d_my += d_means3d_up[gauss_idx * 3 + 1] * inv_vc;
    d_mz += d_means3d_up[gauss_idx * 3 + 2] * inv_vc;

    float d_factor[6];
#pragma unroll
    for (int i = 0; i < 6; ++i)
        d_factor[i] = d_cov_factors_up[gauss_idx * 6 + i] * inv_vc;

    const float d_opacity_cond_raw = d_opacities_up[gauss_idx] * inv_vc;

    // BACKWARD — Opacity: opacity_cond = passes ? opacity_base * wt : 0

    float d_opac_local = 0.0f, d_wt_local = 0.0f;
    if (passes) {
        d_opac_local = d_opacity_cond_raw * wt * sigmoid_grad(opacity_base);
        d_wt_local = d_opacity_cond_raw * opacity_base;
    }

    // BACKWARD — conditioning. Mean and gate through the Schur blocks; the covariance through N.
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

    float d_scales_4d_log_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d_Amat[4][4];
    build_sigma_schur_bwd(A, s, d_cov_3d_marg, d_cov_xt, d_cov_t_marg, d_Amat, d_scales_4d_log_local);
    device_conditioned_factor_bwd(A, s, cov_xt, cov_t_marg, cov_eps, d_factor, d_Amat, d_scales_4d_log_local);

    float d_ql_hat[4], d_qr_hat[4];
    build_A_rot4d_bwd(ql, qr, d_Amat, d_ql_hat, d_qr_hat);

    float d_ql_raw[4], d_qr_raw[4];
    quat_normalize_bwd(ql, d_ql_hat, ql_norm, d_ql_raw);
    quat_normalize_bwd(qr, d_qr_hat, qr_norm, d_qr_raw);

    // ACCUMULATE via atomicAdd

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
 * Host entry: vis_count = bincount(gaussian_ids), then the packed kernel (skipped when nnz == 0).
 *
 * Returns (d_quats_l [N, 4], d_quats_r [N, 4], d_scales_4d_log [N, 4], d_means [N, 3], d_mu_t [N],
 * d_opacities [N], d_sh0 [N, 1, 3], d_shN [N, K_max - 1, 3], d_sh_temporal [N, T, 3] or [0]).
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> ut_4d_fused_packed_projection_bwd(
    const Tensor& d_colors,        // [C, N, 3]
    const Tensor& d_means_3d_cond, // [N, 3]
    const Tensor& d_cov_factors,   // [N, 6]
    const Tensor& d_opacities,     // [N]
    const Tensor& camera_ids,      // [nnz] int64
    const Tensor& gaussian_ids,    // [nnz] int64
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
    const int nnz = (int)camera_ids.size(0);

    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");

    auto opts_f = means.options();
    auto opts_l = means.options().dtype(c10::kLong);

    Tensor d_quats_l_out = at::zeros({N, 4}, opts_f);
    Tensor d_quats_r_out = at::zeros({N, 4}, opts_f);
    Tensor d_scales_4d_log_out = at::zeros({N, 4}, opts_f);
    Tensor d_means_out = at::zeros({N, 3}, opts_f);
    Tensor d_mu_t_out = at::zeros({N}, opts_f);
    Tensor d_opacities_out = at::zeros({N}, opts_f);
    Tensor d_sh0_out = at::zeros(sh0.sizes(), sh0.options()); // contiguous: the kernel indexes it densely
    Tensor d_shN_out = at::zeros(shN.sizes(), shN.options());
    Tensor d_sh_temporal_out = (sh_temporal.numel() > 0) ? at::zeros_like(sh_temporal) : at::zeros({0}, opts_f);

    if (nnz == 0) {
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

    // vis_count[n] = number of visible cameras for Gaussian n = bincount(gaussian_ids)
    Tensor vis_count = at::zeros({N}, opts_l);
    at::bincount_out(
        vis_count,
        gaussian_ids,
        /*weights=*/at::Tensor{},
        /*minlength=*/N
    );

    const float* sh_temporal_ptr = (sh_temporal.numel() > 0) ? sh_temporal.contiguous().data_ptr<float>() : nullptr;
    float* d_sh_temporal_ptr = (d_sh_temporal_out.numel() > 0) ? d_sh_temporal_out.data_ptr<float>() : nullptr;

    const int block_size = 256;
    const int grid_size = (nnz + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();

    ANYCAM4D_DISPATCH_APPEARANCE(
        appearance_kind,
        ut_4d_fused_packed_projection_bwd_kernel<M><<<grid_size, block_size, 0, stream>>>(
            d_colors.contiguous().data_ptr<float>(),
            d_means_3d_cond.contiguous().data_ptr<float>(),
            d_cov_factors.contiguous().data_ptr<float>(),
            d_opacities.contiguous().data_ptr<float>(),
            camera_ids.contiguous().data_ptr<int64_t>(),
            gaussian_ids.contiguous().data_ptr<int64_t>(),
            vis_count.data_ptr<int64_t>(),
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
            (int)viewmats.size(0),
            N,
            tau,
            cov_eps,
            prefilter_threshold,
            nnz,
            d_quats_l_out.data_ptr<float>(),
            d_quats_r_out.data_ptr<float>(),
            d_scales_4d_log_out.data_ptr<float>(),
            d_means_out.data_ptr<float>(),
            d_mu_t_out.data_ptr<float>(),
            d_opacities_out.data_ptr<float>(),
            d_sh0_out.data_ptr<float>(),
            d_shN_out.data_ptr<float>(),
            d_sh_temporal_ptr
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
