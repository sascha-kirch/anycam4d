/**
 * Fused 4D conditioning + UT projection + 4D SH forward, one thread per (Gaussian, camera) pair.
 *
 * Outputs: means2d [C, N, 2], conics [C, N, 3], depths [C, N] (sort key ‖μ_c‖), radii [C, N, 2] for tile
 * assignment only; means_3d_cond [N, 3], cov_factors [N, 6], opacities [N], colors [C, N, 3] carry gradients;
 * wt [N] is a side output. UT compensations are discarded.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

#include "conditioning.cuh"
#include "sh4d.cuh"
#include "ut_4d.cuh"

using at::Tensor;

template <ProjectionFamily F, AppearanceKind M>
__launch_bounds__(256, 2) __global__ void ut_4d_fused_projection_fwd_kernel(
    // 4D Gaussian parameters [N, ...]
    const float* __restrict__ quats_l,       // [N, 4] wxyz, unnormalized
    const float* __restrict__ quats_r,       // [N, 4] wxyz, unnormalized
    const float* __restrict__ scales_4d_log, // [N, 4] log-space
    const float* __restrict__ means,         // [N, 3]
    const float* __restrict__ mu_t,          // [N]
    const float* __restrict__ opacities_in,  // [N] raw logit
    // SH coefficients
    const float* __restrict__ sh0,         // [N, 1, 3]
    const float* __restrict__ shN,         // [N, K_max - 1, 3]
    const float* __restrict__ sh_temporal, // [N, T, 3] temporal coefficients, or NULL
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    // Camera view (POD, by value)
    CameraViewDevice cam,
    // Scalar parameters
    float tau,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    // Per-camera outputs [C, N, ...]
    float* __restrict__ means2d_out, // [C, N, 2]
    float* __restrict__ conics_out,  // [C, N, 3]
    float* __restrict__ depths_out,  // [C, N]
    int32_t* __restrict__ radii_out, // [C, N, 2]
    float* __restrict__ colors_out,  // [C, N, 3]
    // Per-Gaussian outputs [N, ...] (written by cam_idx == 0 thread only)
    float* __restrict__ means3d_out,     // [N, 3]
    float* __restrict__ cov_factors_out, // [N, 6]
    float* __restrict__ opacities_out,   // [N]
    float* __restrict__ wt_out,          // [N]
    int N
)
{
    const int C = cam.C;
    const int image_width = cam.width;
    const int image_height = cam.height;
    const float* viewmats = cam.viewmats;
    const float* Ks = cam.Ks;

    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= C * N)
        return;

    const int gauss_idx = global_idx % N;
    const int cam_idx = global_idx / N;
    const int out_idx = cam_idx * N + gauss_idx;

    // STAGE 1: Temporal Conditioning

    float mx, my, mz, cov_L[6], wt, opacity_cond;
    device_condition_rot4d<float>(
        quats_l + gauss_idx * 4,
        quats_r + gauss_idx * 4,
        scales_4d_log + gauss_idx * 4,
        means[gauss_idx * 3 + 0],
        means[gauss_idx * 3 + 1],
        means[gauss_idx * 3 + 2],
        mu_t[gauss_idx],
        opacities_in[gauss_idx],
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

    // Per-Gaussian side outputs: written by cam_idx == 0 only
    if (cam_idx == 0) {
        means3d_out[gauss_idx * 3 + 0] = mx;
        means3d_out[gauss_idx * 3 + 1] = my;
        means3d_out[gauss_idx * 3 + 2] = mz;
#pragma unroll
        for (int i = 0; i < 6; ++i)
            cov_factors_out[gauss_idx * 6 + i] = cov_L[i];
        opacities_out[gauss_idx] = opacity_cond;
        wt_out[gauss_idx] = wt;
    }

    // STAGE 2: UT sigma-point projection (tile-assignment outputs)

    // Default radii = 0 if culled at any point
    if (opacity_cond < ALPHA_THRESHOLD) {
        radii_out[out_idx * 2 + 0] = 0;
        radii_out[out_idx * 2 + 1] = 0;
        // Still need to evaluate SH (stage 3) so the colors tensor is populated.
        // Skipping the projection block, fall through to SH eval.
    }

    bool visible = false;
    if (opacity_cond >= ALPHA_THRESHOLD) {

        // UT weights + spread are paper-frozen constexpr (ut_constants.cuh).
        float wm[7] = {UT_WM0, UT_WMI, UT_WMI, UT_WMI, UT_WMI, UT_WMI, UT_WMI};
        float wc[7] = {UT_WC0, UT_WCI, UT_WCI, UT_WCI, UT_WCI, UT_WCI, UT_WCI};

        const float mu[3] = {mx, my, mz};
        float pts[7][3];
        device_sigma_points_from_factor(mu, cov_L, UT_C_SPREAD, pts);

        const float* vm = viewmats + cam_idx * 16;
        const float* K_ptr = Ks + cam_idx * 9;
        const float fx = K_ptr[0], fy = K_ptr[4], cx_k = K_ptr[2], cy_k = K_ptr[5];

        // Per-camera distortion pointer; family slot count is a compile-time constexpr.
        const float* dist = cam.distortion + cam_idx * CameraOps<F>::n_slots;

        float mean2d_x, mean2d_y;
        float conic_a, conic_b, conic_c;
        float depth_val;
        int32_t radius_x, radius_y;

        visible = device_ut_project<F>(
            pts,
            cov_L,
            wm,
            wc,
            vm,
            fx,
            fy,
            cx_k,
            cy_k,
            eps2d,
            opacity_cond,
            near_plane,
            far_plane,
            image_width,
            image_height,
            dist,
            mean2d_x,
            mean2d_y,
            conic_a,
            conic_b,
            conic_c,
            depth_val,
            radius_x,
            radius_y
        );

        if (visible) {
            means2d_out[out_idx * 2 + 0] = mean2d_x;
            means2d_out[out_idx * 2 + 1] = mean2d_y;
            conics_out[out_idx * 3 + 0] = conic_a;
            conics_out[out_idx * 3 + 1] = conic_b;
            conics_out[out_idx * 3 + 2] = conic_c;
            depths_out[out_idx] = depth_val;
            radii_out[out_idx * 2 + 0] = radius_x;
            radii_out[out_idx * 2 + 1] = radius_y;
        }
        else {
            radii_out[out_idx * 2 + 0] = 0;
            radii_out[out_idx * 2 + 1] = 0;
        }
    }

    // ====================================================================
    // STAGE 3: 4D SH Color Evaluation
    // Always run so colors[C, N, 3] is fully populated (and differentiable).
    // ====================================================================

    const float* vm = viewmats + cam_idx * 16;
    const float tx = vm[3], ty = vm[7], tz = vm[11];
    const float cam_x = -(vm[0] * tx + vm[4] * ty + vm[8] * tz);
    const float cam_y = -(vm[1] * tx + vm[5] * ty + vm[9] * tz);
    const float cam_z = -(vm[2] * tx + vm[6] * ty + vm[10] * tz);

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

    colors_out[out_idx * 3 + 0] = r;
    colors_out[out_idx * 3 + 1] = g;
    colors_out[out_idx * 3 + 2] = b;
}

/**
 * Host entry; an empty sh_temporal runs spatial SH only.
 * Returns (means2d, conics, depths, radii, colors, means_3d_cond, cov_factors, opacities, wt).
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> ut_4d_fused_projection_fwd(
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
    const Tensor& radial_coeffs,
    float tau,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    int image_width,
    int image_height,
    int family
)
{
    const int N = (int)means.size(0);
    const int C = (int)viewmats.size(0);
    const int total = C * N;

    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");

    // Defense-in-depth: fixed-width [C, P] with the family slot count (4 fisheye / 12
    // pinhole-family). The host switch's exhaustive default is the only model gate.
    const int expected_stride = (family == FISHEYE_FAMILY) ? 4 : 12;
    TORCH_CHECK(
        radial_coeffs.dim() == 2 && radial_coeffs.size(0) == C && radial_coeffs.size(1) == expected_stride,
        "radial_coeffs must be [C, ",
        expected_stride,
        "] for this projection family; got a tensor of dim ",
        radial_coeffs.dim()
    );

    auto opts_f = means.options();
    auto opts_i = means.options().dtype(c10::kInt);

    // Per-camera (tile-assignment) outputs
    Tensor means2d_out = at::zeros({C, N, 2}, opts_f);
    Tensor conics_out = at::zeros({C, N, 3}, opts_f);
    Tensor depths_out = at::zeros({C, N}, opts_f);
    Tensor radii_out = at::zeros({C, N, 2}, opts_i);

    // Per-camera differentiable color output
    Tensor colors_out = at::zeros({C, N, 3}, opts_f);

    // Per-Gaussian differentiable outputs
    Tensor means3d_out = at::zeros({N, 3}, opts_f);
    Tensor cov_factors_out = at::zeros({N, 6}, opts_f);
    Tensor opacities_out = at::zeros({N}, opts_f);
    Tensor wt_out = at::zeros({N}, opts_f);

    if (total == 0) {
        return {
            means2d_out,
            conics_out,
            depths_out,
            radii_out,
            colors_out,
            means3d_out,
            cov_factors_out,
            opacities_out,
            wt_out
        };
    }

    const float* sh_temporal_ptr = (sh_temporal.numel() > 0) ? sh_temporal.contiguous().data_ptr<float>() : nullptr;
    // Camera tensors are guaranteed contiguous by Python (to_kernel_args); verify instead of copy.
    TORCH_CHECK(
        viewmats.is_contiguous() && Ks.is_contiguous() && radial_coeffs.is_contiguous(),
        "camera tensors must be contiguous"
    );

    CameraViewDevice cam{
        viewmats.data_ptr<float>(),
        Ks.data_ptr<float>(),
        radial_coeffs.data_ptr<float>(),
        C,
        (int)radial_coeffs.size(1),
        image_width,
        image_height
    };

    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_UT_FUSED(FAMILY)                                                                                        \
    ut_4d_fused_projection_fwd_kernel<FAMILY, M><<<grid_size, block_size, 0, stream>>>(                                \
        quats_l.contiguous().data_ptr<float>(),                                                                        \
        quats_r.contiguous().data_ptr<float>(),                                                                        \
        scales_4d_log.contiguous().data_ptr<float>(),                                                                  \
        means.contiguous().data_ptr<float>(),                                                                          \
        mu_t.contiguous().data_ptr<float>(),                                                                           \
        opacities.contiguous().data_ptr<float>(),                                                                      \
        sh0.contiguous().data_ptr<float>(),                                                                            \
        shN.contiguous().data_ptr<float>(),                                                                            \
        sh_temporal_ptr,                                                                                               \
        sh_degree,                                                                                                     \
        (int)shN.size(1) * 3,                                                                                          \
        n_temporal_terms,                                                                                              \
        temporal_stride,                                                                                               \
        cam,                                                                                                           \
        tau,                                                                                                           \
        cov_eps,                                                                                                       \
        eps2d,                                                                                                         \
        near_plane,                                                                                                    \
        far_plane,                                                                                                     \
        prefilter_threshold,                                                                                           \
        means2d_out.data_ptr<float>(),                                                                                 \
        conics_out.data_ptr<float>(),                                                                                  \
        depths_out.data_ptr<float>(),                                                                                  \
        radii_out.data_ptr<int32_t>(),                                                                                 \
        colors_out.data_ptr<float>(),                                                                                  \
        means3d_out.data_ptr<float>(),                                                                                 \
        cov_factors_out.data_ptr<float>(),                                                                             \
        opacities_out.data_ptr<float>(),                                                                               \
        wt_out.data_ptr<float>(),                                                                                      \
        N                                                                                                              \
    )

    switch (family) {
    case PINHOLE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_UT_FUSED(PINHOLE_FAMILY));
        break;
    case FISHEYE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_UT_FUSED(FISHEYE_FAMILY));
        break;
    default:
        TORCH_CHECK(false, "unsupported projection family ", family);
    }
#undef LAUNCH_UT_FUSED

    return {
        means2d_out,
        conics_out,
        depths_out,
        radii_out,
        colors_out,
        means3d_out,
        cov_factors_out,
        opacities_out,
        wt_out
    };
}
