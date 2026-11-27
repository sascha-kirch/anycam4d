/**
 * Fused 4D conditioning + SH + EWA projection forward, one thread per (Gaussian, camera) pair;
 * gsplat's isect_tiles and rasterize_to_pixels complete the render.
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

// π as a float literal
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/**
 * Fused forward kernel: 4D conditioning → 4D SH → world-to-camera → EWA projection.
 *
 * Grid: [ceil(C*N / block_size)] blocks.  One thread per (Gaussian, Camera) pair.
 *   gauss_idx = global_idx % N
 *   cam_idx   = global_idx / N
 *
 * Side outputs (means3d_out, wt_out) are written only by the thread with cam_idx == 0.
 */
template <ProjectionFamily F, bool kLens, AppearanceKind M>
__launch_bounds__(256, 2) __global__ void ewa_4d_fused_conditioned_projection_fwd_kernel(
    // 4D Gaussian parameters [N, ...]
    const float* __restrict__ quats_l,       // [N, 4] wxyz, unnormalized
    const float* __restrict__ quats_r,       // [N, 4] wxyz, unnormalized
    const float* __restrict__ scales_4d_log, // [N, 4] log-space
    const float* __restrict__ means,         // [N, 3]
    const float* __restrict__ mu_t,          // [N]
    const float* __restrict__ opacities,     // [N] raw logit
    // SH coefficients
    const float* __restrict__ sh0,         // [N, 1, 3]
    const float* __restrict__ shN,         // [N, K_max - 1, 3]
    const float* __restrict__ sh_temporal, // [N, T, 3] temporal coefficients, or NULL
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    // Camera parameters [C, ...]
    const float* __restrict__ viewmats,   // [C, 4, 4] row-major
    const float* __restrict__ Ks,         // [C, 3, 3] row-major
    const float* __restrict__ dist,       // [C, n_slots] lens distortion; NULL when !kLens
    const float* __restrict__ clamp_lims, // [C, 4] Brown–Conrady clamp limits; NULL otherwise
    int C,
    // Scalar parameters
    float tau,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    int image_width,
    int image_height,
    bool calc_compensations,
    // Dense outputs [C, N, ...]
    float* __restrict__ means2d_out, // [C, N, 2]
    float* __restrict__ conics_out,  // [C, N, 3]
    float* __restrict__ colors_out,  // [C, N, 3]
    float* __restrict__ opac_out,    // [C, N]
    float* __restrict__ depths_out,  // [C, N]
    int32_t* __restrict__ radii_out, // [C, N, 2]
    // Per-Gaussian side outputs (written by cam_idx==0 thread only)
    float* __restrict__ means3d_out, // [N, 3]
    float* __restrict__ wt_out,      // [N]
    int N
)
{
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= C * N)
        return;

    const int gauss_idx = global_idx % N;
    const int cam_idx = global_idx / N;
    const int out_idx = cam_idx * N + gauss_idx; // flat index in [C, N] layout

    // STAGE 1: Temporal Conditioning (double-quaternion Schur complement)

    float mx, my, mz, cov_L[6], wt, opacity_cond;
    device_condition_rot4d<float>(
        quats_l + gauss_idx * 4,
        quats_r + gauss_idx * 4,
        scales_4d_log + gauss_idx * 4,
        means[gauss_idx * 3 + 0],
        means[gauss_idx * 3 + 1],
        means[gauss_idx * 3 + 2],
        mu_t[gauss_idx],
        opacities[gauss_idx],
        tau,
        cov_eps,
        prefilter_threshold,
        mx,
        my,
        mz,
        cov_L,
        wt,
        opacity_cond
    );

    // Side outputs: written once per Gaussian (cam_idx == 0)
    if (cam_idx == 0) {
        means3d_out[gauss_idx * 3 + 0] = mx;
        means3d_out[gauss_idx * 3 + 1] = my;
        means3d_out[gauss_idx * 3 + 2] = mz;
        wt_out[gauss_idx] = wt;
    }

    // The projection's opacity cull, taken before any per-camera work: skips temporally gated Gaussians.
    if (opacity_cond < ALPHA_THRESHOLD) {
        radii_out[out_idx * 2 + 0] = 0;
        radii_out[out_idx * 2 + 1] = 0;
        return;
    }

    const float* vm = viewmats + cam_idx * 16;
    const float tx = vm[3], ty = vm[7], tz = vm[11];

    // STAGE 2: World-to-Camera Transform

    // Row-major rotation rows from viewmat
    const float r00 = vm[0], r01 = vm[1], r02 = vm[2];
    const float r10 = vm[4], r11 = vm[5], r12 = vm[6];
    const float r20 = vm[8], r21 = vm[9], r22 = vm[10];

    // Transform conditioned mean to camera space: mean_c = R_w2c @ mean_world + t_w2c
    const float mc_x = r00 * mx + r01 * my + r02 * mz + tx;
    const float mc_y = r10 * mx + r11 * my + r12 * mz + ty;
    const float mc_z = r20 * mx + r21 * my + r22 * mz + tz;

    // Rotate 3D covariance to camera space: covar_c = R @ cov_3d_cond @ R^T
    float cc00, cc01, cc02, cc11, cc12, cc22;
    float cov_3d_cond[6];
    device_gram_lower(cov_L, cov_3d_cond);
    device_covarW2C(r00, r01, r02, r10, r11, r12, r20, r21, r22, cov_3d_cond, cc00, cc01, cc02, cc11, cc12, cc22);

    // STAGE 3: EWA Perspective Projection + Culling

    // Camera intrinsics (row-major [3, 3])
    const float* K_ptr = Ks + cam_idx * 9;
    const float fx = K_ptr[0];
    const float fy = K_ptr[4];
    const float cx = K_ptr[2];
    const float cy = K_ptr[5];

    float mean2d_x, mean2d_y;
    float conic_a, conic_b, conic_c;
    float depth_out;
    int32_t radius_x, radius_y;
    float compensation;

    const bool visible = device_ewa_project<F, kLens>(
        mc_x,
        mc_y,
        mc_z,
        cc00,
        cc01,
        cc02,
        cc11,
        cc12,
        cc22,
        fx,
        fy,
        cx,
        cy,
        device_ewa_lens<F, kLens>(dist, clamp_lims, cam_idx),
        eps2d,
        opacity_cond,
        near_plane,
        far_plane,
        image_width,
        image_height,
        calc_compensations,
        mean2d_x,
        mean2d_y,
        conic_a,
        conic_b,
        conic_c,
        depth_out,
        radius_x,
        radius_y,
        compensation
    );

    if (!visible) {
        radii_out[out_idx * 2 + 0] = 0;
        radii_out[out_idx * 2 + 1] = 0;
        return;
    }

    // STAGE 4: SH colour, visible pairs only
    const float cam_x = -(vm[0] * tx + vm[4] * ty + vm[8] * tz);
    const float cam_y = -(vm[1] * tx + vm[5] * ty + vm[9] * tz);
    const float cam_z = -(vm[2] * tx + vm[6] * ty + vm[10] * tz);

    // Gaussian → camera direction (unnormalized; device_sh_eval_channel normalizes)
    const float dir_x = mx - cam_x;
    const float dir_y = my - cam_y;
    const float dir_z = mz - cam_z;
    const float dir_t = tau - mu_t[gauss_idx];
    float r, g, b;
    AppearanceOps<M>::eval(
        sh_degree,
        n_temporal_terms,
        temporal_stride,
        dir_x,
        dir_y,
        dir_z,
        dir_t,
        ShCoeffs{sh0 + gauss_idx * 3, shN + gauss_idx * shN_stride},
        (sh_temporal != nullptr && n_temporal_terms > 0) ? sh_temporal + gauss_idx * temporal_stride : nullptr,
        r,
        g,
        b
    );

    // Write outputs
    means2d_out[out_idx * 2 + 0] = mean2d_x;
    means2d_out[out_idx * 2 + 1] = mean2d_y;
    conics_out[out_idx * 3 + 0] = conic_a;
    conics_out[out_idx * 3 + 1] = conic_b;
    conics_out[out_idx * 3 + 2] = conic_c;
    colors_out[out_idx * 3 + 0] = r;
    colors_out[out_idx * 3 + 1] = g;
    colors_out[out_idx * 3 + 2] = b;
    opac_out[out_idx] = opacity_cond * compensation;
    depths_out[out_idx] = depth_out;
    radii_out[out_idx * 2 + 0] = radius_x;
    radii_out[out_idx * 2 + 1] = radius_y;
}

/**
 * Launch ewa_4d_fused_conditioned_projection_fwd_kernel and return output tensors.
 *
 * All inputs are made contiguous before launch.  Output tensors are zero-initialized
 * on the same device as `means` — culled Gaussians retain zeros in all outputs.
 *
 * @param sh_temporal  May be an empty tensor (numel() == 0) to indicate no temporal SH.
 *                     The kernel skips temporal evaluation when n_temporal_terms == 0.
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
launch_ewa_4d_fused_conditioned_projection_fwd_kernel(
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
    TORCH_CHECK(N > 0 && C > 0, "N and C must be positive");

    auto opts_f = means.options();                  // float32, same device
    auto opts_i = means.options().dtype(c10::kInt); // int32,   same device

    // Allocate zero-initialized output tensors
    Tensor means2d_out = at::zeros({C, N, 2}, opts_f);
    Tensor conics_out = at::zeros({C, N, 3}, opts_f);
    Tensor colors_out = at::zeros({C, N, 3}, opts_f);
    Tensor opac_out = at::zeros({C, N}, opts_f);
    Tensor depths_out = at::zeros({C, N}, opts_f);
    Tensor radii_out = at::zeros({C, N, 2}, opts_i);
    Tensor means3d_out = at::zeros({N, 3}, opts_f);
    Tensor wt_out = at::zeros({N}, opts_f);

    if (total == 0) {
        return {means2d_out, conics_out, colors_out, opac_out, depths_out, radii_out, means3d_out, wt_out};
    }

    const float* sh_temporal_ptr = (sh_temporal.numel() > 0) ? sh_temporal.contiguous().data_ptr<float>() : nullptr;

    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();

    ANYCAM4D_DISPATCH_EWA_LENS(
        lens_args.has_lens,
        family,
        ANYCAM4D_DISPATCH_APPEARANCE(
            appearance_kind,
            ewa_4d_fused_conditioned_projection_fwd_kernel<F, kLens, M><<<grid_size, block_size, 0, stream>>>(
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
                means2d_out.data_ptr<float>(),
                conics_out.data_ptr<float>(),
                colors_out.data_ptr<float>(),
                opac_out.data_ptr<float>(),
                depths_out.data_ptr<float>(),
                radii_out.data_ptr<int32_t>(),
                means3d_out.data_ptr<float>(),
                wt_out.data_ptr<float>(),
                N
            )
        )
    );

    return {means2d_out, conics_out, colors_out, opac_out, depths_out, radii_out, means3d_out, wt_out};
}
