/**
 * Packed fused UT forward: conditioning + UT projection + 4D SH, in the two-pass scheme of the
 * EWA packed kernel (pass 1 counts per block, the host cumsums and allocates, pass 2 scatters).
 *
 * Packed [nnz]: camera_ids, gaussian_ids, means2d, conics, depths, radii, and indptr [C + 1].
 * Dense and camera-independent, written in pass 1 by cam_idx == 0: means_3d_cond, cov_factors, opacities, wt.
 * Dense colors [C, N, 3] are written in pass 2 for visible pairs only; the rasterizer reads no others.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>

#include "conditioning.cuh"
#include "sh4d.cuh"
#include "ut_4d.cuh"

using at::Tensor;

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

#define N_THREADS_PACKED_UT 256

/**
 * Shared kernel for both passes: conditioning, UT projection (decides visibility), SH colour.
 */
template <ProjectionFamily F, AppearanceKind M>
__launch_bounds__(N_THREADS_PACKED_UT) __global__ void ut_4d_fused_packed_projection_fwd_kernel(
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
    const float* __restrict__ sh_temporal, // [N, T, 3] or NULL
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    // Camera view (POD, by value)
    CameraViewDevice cam,
    int N,
    // Scalar parameters
    float tau,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    // Two-pass packing controls (one of each pair is null depending on the pass)
    const int32_t* __restrict__ block_accum, // [C * blocks_per_row] — null on pass 1
    int32_t* __restrict__ block_cnts,        // [C * blocks_per_row] — null on pass 2
    // Pass 2 packed outputs (all NULL on pass 1)
    int32_t* __restrict__ indptr,       // [C + 1]  CSR row pointers
    int64_t* __restrict__ camera_ids,   // [nnz]
    int64_t* __restrict__ gaussian_ids, // [nnz]
    float* __restrict__ means2d_out,    // [nnz, 2]
    float* __restrict__ conics_out,     // [nnz, 3]
    float* __restrict__ depths_out,     // [nnz]
    int32_t* __restrict__ radii_out,    // [nnz, 2] int32
    // Dense outputs
    // colors_out [C, N, 3]: written in Pass 2 for visible (cam, gauss) pairs
    float* __restrict__ colors_out, // [C, N, 3] — NULL on pass 1
    // Side outputs [N, ...]: written in Pass 1 by cam_idx == 0 threads
    float* __restrict__ means3d_out,     // [N, 3]
    float* __restrict__ cov_factors_out, // [N, 6]
    float* __restrict__ opacities_out,   // [N]
    float* __restrict__ wt_out           // [N]
)
{
    const int C = cam.C;
    const int image_width = cam.width;
    const int image_height = cam.height;
    const float* viewmats = cam.viewmats;
    const float* Ks = cam.Ks;

    const int32_t blocks_per_row = gridDim.x;
    const int32_t block_col_idx = blockIdx.x;
    const int32_t cam_idx = blockIdx.y;
    const int32_t block_idx = cam_idx * blocks_per_row + block_col_idx;
    const int32_t gauss_idx = block_col_idx * blockDim.x + threadIdx.x;

    bool valid = (cam_idx < C) && (gauss_idx < N);

    // STAGE 1: Temporal Conditioning

    float mx, my, mz, cov_L[6], wt, opacity_cond;
    if (valid) {
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

        // Dense side outputs: written once per Gaussian during Pass 1
        if (block_cnts != nullptr && cam_idx == 0) {
            means3d_out[gauss_idx * 3 + 0] = mx;
            means3d_out[gauss_idx * 3 + 1] = my;
            means3d_out[gauss_idx * 3 + 2] = mz;
#pragma unroll
            for (int i = 0; i < 6; ++i)
                cov_factors_out[gauss_idx * 6 + i] = cov_L[i];
            opacities_out[gauss_idx] = opacity_cond;
            wt_out[gauss_idx] = wt;
        }
    }

    // STAGE 2: UT sigma-point projection (determines visibility)

    float mean2d_x = 0.f, mean2d_y = 0.f;
    float conic_a = 0.f, conic_b = 0.f, conic_c = 0.f;
    float depth_val = 0.f;
    int32_t radius_x = 0, radius_y = 0;

    if (valid) {
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

            valid = device_ut_project<F>(
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
        }
        else {
            valid = false;
        }
    }

    // Two-pass packing

    int32_t thread_data = static_cast<int32_t>(valid);

    if (block_cnts != nullptr) {
        // --- Pass 1: count visible per block ---
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, N_THREADS_PACKED_UT> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        }
        else {
            aggregate = 0;
        }
        if (threadIdx.x == 0) {
            block_cnts[block_idx] = aggregate;
        }
    }
    else {
        // --- Pass 2: scatter to packed positions + write dense colors ---

        // STAGE 3: 4D SH Color Evaluation (visible pairs only)
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (valid) {
            const float* vm = viewmats + cam_idx * 16;
            const float tx = vm[3], ty = vm[7], tz = vm[11];
            const float cam_x = -(vm[0] * tx + vm[4] * ty + vm[8] * tz);
            const float cam_y = -(vm[1] * tx + vm[5] * ty + vm[9] * tz);
            const float cam_z = -(vm[2] * tx + vm[6] * ty + vm[10] * tz);

            const float dir_x = mx - cam_x;
            const float dir_y = my - cam_y;
            const float dir_z = mz - cam_z;
            const float dir_t = tau - mu_t[gauss_idx];
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

            // Write dense colors for RasterizeWorldFunction
            const int32_t color_idx = (cam_idx * N + gauss_idx) * 3;
            colors_out[color_idx + 0] = r;
            colors_out[color_idx + 1] = g;
            colors_out[color_idx + 2] = b;
        }

        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, N_THREADS_PACKED_UT> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }

        if (valid) {
            if (block_idx > 0) {
                thread_data += block_accum[block_idx - 1];
            }
            // Write packed tile-assignment outputs
            camera_ids[thread_data] = cam_idx;
            gaussian_ids[thread_data] = gauss_idx;
            means2d_out[thread_data * 2 + 0] = mean2d_x;
            means2d_out[thread_data * 2 + 1] = mean2d_y;
            conics_out[thread_data * 3 + 0] = conic_a;
            conics_out[thread_data * 3 + 1] = conic_b;
            conics_out[thread_data * 3 + 2] = conic_c;
            depths_out[thread_data] = depth_val;
            radii_out[thread_data * 2 + 0] = radius_x;
            radii_out[thread_data * 2 + 1] = radius_y;
        }

        // Lane 0 of first block in each camera row writes indptr
        if (threadIdx.x == 0 && block_col_idx == 0) {
            if (cam_idx == 0) {
                indptr[0] = 0;
                indptr[C] = block_accum[C * blocks_per_row - 1];
            }
            else {
                indptr[cam_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

/**
 * Host entry for the packed UT forward.
 *
 * Returns (camera_ids [nnz], gaussian_ids [nnz], indptr [C+1], means2d [nnz, 2], conics [nnz, 3],
 * depths [nnz], radii [nnz, 2]) for tile assignment, then dense colors [C, N, 3], means_3d_cond [N, 3],
 * cov_factors [N, 6], opacities [N] (differentiable) and wt [N].
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
ut_4d_fused_packed_projection_fwd(
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

    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(N > 0 && C > 0, "N and C must be positive");

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
    auto opts_l = means.options().dtype(c10::kLong);

    const int blocks_per_row = (N + N_THREADS_PACKED_UT - 1) / N_THREADS_PACKED_UT;
    const int total_blocks = C * blocks_per_row;
    dim3 threads(N_THREADS_PACKED_UT);
    dim3 grid(blocks_per_row, C);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Dense side outputs (always allocated, camera-independent)
    Tensor means3d_out = at::zeros({N, 3}, opts_f);
    Tensor cov_factors_out = at::zeros({N, 6}, opts_f);
    Tensor opacities_out = at::zeros({N}, opts_f);
    Tensor wt_out = at::zeros({N}, opts_f);

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

    // --- Pass 1: count visible per block ---
    Tensor block_cnts = at::empty({total_blocks}, opts_i);

#define LAUNCH_PACKED_PASS1(FAMILY)                                                                                    \
    ut_4d_fused_packed_projection_fwd_kernel<FAMILY, M><<<grid, threads, 0, stream>>>(                                 \
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
        N,                                                                                                             \
        tau,                                                                                                           \
        cov_eps,                                                                                                       \
        eps2d,                                                                                                         \
        near_plane,                                                                                                    \
        far_plane,                                                                                                     \
        prefilter_threshold,                                                                                           \
        nullptr,                        /* block_accum (pass 1) */                                                     \
        block_cnts.data_ptr<int32_t>(), /* block_cnts */                                                               \
        nullptr,                        /* indptr */                                                                   \
        nullptr,                        /* camera_ids */                                                               \
        nullptr,                        /* gaussian_ids */                                                             \
        nullptr,                        /* means2d */                                                                  \
        nullptr,                        /* conics */                                                                   \
        nullptr,                        /* depths */                                                                   \
        nullptr,                        /* radii */                                                                    \
        nullptr,                        /* colors_out (pass 1: not needed) */                                          \
        means3d_out.data_ptr<float>(),                                                                                 \
        cov_factors_out.data_ptr<float>(),                                                                             \
        opacities_out.data_ptr<float>(),                                                                               \
        wt_out.data_ptr<float>()                                                                                       \
    )

    switch (family) {
    case PINHOLE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_PACKED_PASS1(PINHOLE_FAMILY));
        break;
    case FISHEYE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_PACKED_PASS1(FISHEYE_FAMILY));
        break;
    default:
        TORCH_CHECK(false, "unsupported projection family ", family);
    }
#undef LAUNCH_PACKED_PASS1

    // Cumsum → global offsets
    Tensor block_accum = at::cumsum(block_cnts, 0, c10::kInt);
    int32_t nnz = block_accum[-1].item<int32_t>();

    // Allocate packed output tensors
    Tensor camera_ids_out = at::empty({nnz}, opts_l);
    Tensor gaussian_ids_out = at::empty({nnz}, opts_l);
    Tensor indptr_out = at::empty({C + 1}, opts_i);
    Tensor means2d_out = at::empty({nnz, 2}, opts_f);
    Tensor conics_out = at::empty({nnz, 3}, opts_f);
    Tensor depths_out = at::empty({nnz}, opts_f);
    Tensor radii_out = at::empty({nnz, 2}, opts_i);

    // Dense colors tensor: always allocated [C, N, 3]
    Tensor colors_out = at::zeros({C, N, 3}, opts_f);

    if (nnz > 0) {
        // --- Pass 2: scatter packed outputs + write dense colors ---
#define LAUNCH_PACKED_PASS2(FAMILY)                                                                                    \
    ut_4d_fused_packed_projection_fwd_kernel<FAMILY, M><<<grid, threads, 0, stream>>>(                                 \
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
        N,                                                                                                             \
        tau,                                                                                                           \
        cov_eps,                                                                                                       \
        eps2d,                                                                                                         \
        near_plane,                                                                                                    \
        far_plane,                                                                                                     \
        prefilter_threshold,                                                                                           \
        block_accum.data_ptr<int32_t>(), /* block_accum (pass 2) */                                                    \
        nullptr,                         /* block_cnts (pass 2) */                                                     \
        indptr_out.data_ptr<int32_t>(),                                                                                \
        camera_ids_out.data_ptr<int64_t>(),                                                                            \
        gaussian_ids_out.data_ptr<int64_t>(),                                                                          \
        means2d_out.data_ptr<float>(),                                                                                 \
        conics_out.data_ptr<float>(),                                                                                  \
        depths_out.data_ptr<float>(),                                                                                  \
        radii_out.data_ptr<int32_t>(),                                                                                 \
        colors_out.data_ptr<float>(),                                                                                  \
        nullptr, /* means_3d_cond (already written in pass 1) */                                                       \
        nullptr, /* cov_factors (already written in pass 1) */                                                         \
        nullptr, /* opacities (already written in pass 1) */                                                           \
        nullptr  /* wt (already written in pass 1) */                                                                  \
    )

        switch (family) {
        case PINHOLE_FAMILY:
            ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_PACKED_PASS2(PINHOLE_FAMILY));
            break;
        case FISHEYE_FAMILY:
            ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_PACKED_PASS2(FISHEYE_FAMILY));
            break;
        default:
            TORCH_CHECK(false, "unsupported projection family ", family);
        }
#undef LAUNCH_PACKED_PASS2
    }
    else {
        indptr_out.fill_(0);
    }

    return {
        camera_ids_out,
        gaussian_ids_out,
        indptr_out,
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
