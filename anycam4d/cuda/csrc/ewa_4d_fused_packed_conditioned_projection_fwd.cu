/**
 * Packed fused 4D conditioning + SH + EWA projection, forward.
 *
 * Kernels cannot allocate, so packing takes two passes of one kernel:
 *   Pass 1 (block_cnts set): count visible pairs per block with cub::BlockReduce; write the
 *          camera-independent side outputs means_3d_cond and wt from cam_idx == 0.
 *   Host:  block_accum = cumsum(block_cnts), nnz = block_accum[last]; allocate [nnz, ...].
 *   Pass 2 (block_accum set): recompute, take each thread's offset from cub::BlockScan
 *          ExclusiveSum, and scatter to local offset + block_accum[block_idx - 1].
 * Recomputing avoids dense [C, N, ...] intermediates.
 *
 * Grid: dim3(ceil(N / 256), C); cam_idx = blockIdx.y, gauss_idx = blockIdx.x * blockDim.x + threadIdx.x.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>

#include "conditioning.cuh"
#include "ewa_4d.cuh"
#include "ewa_4d_lens.cuh"
#include "ewa_4d_lens_limits.cuh"
#include "sh4d.cuh"

using at::Tensor;

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

#define N_THREADS_PACKED 256

/**
 * Shared kernel for both passes; per (camera, Gaussian): conditioning, world-to-camera, EWA projection,
 * and on pass 2 the SH colour of visible pairs. `valid` becomes the projection's visibility and feeds CUB as 0/1.
 * indptr[c] (CSR row pointer) is written by thread 0 of each camera row's first block.
 * Outputs are null on pass 1 except the dense [N] side outputs, which are null on pass 2.
 */
template <ProjectionFamily F, bool kLens, AppearanceKind M>
__launch_bounds__(N_THREADS_PACKED) __global__ void ewa_4d_fused_packed_conditioned_projection_fwd_kernel(
    // 4D Gaussian parameters [N, ...]
    const float* __restrict__ quats_l,       // [N, 4]
    const float* __restrict__ quats_r,       // [N, 4]
    const float* __restrict__ scales_4d_log, // [N, 4]
    const float* __restrict__ means,         // [N, 3]
    const float* __restrict__ mu_t,          // [N]
    const float* __restrict__ opacities,     // [N]
    // SH coefficients
    const float* __restrict__ sh0,         // [N, 1, 3]
    const float* __restrict__ shN,         // [N, K_max - 1, 3]
    const float* __restrict__ sh_temporal, // [N, T, 3] or NULL
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    // Camera parameters [C, ...]
    const float* __restrict__ viewmats,   // [C, 4, 4]
    const float* __restrict__ Ks,         // [C, 3, 3]
    const float* __restrict__ dist,       // [C, n_slots] lens distortion; NULL when !kLens
    const float* __restrict__ clamp_lims, // [C, 4] Brown–Conrady clamp limits; NULL otherwise
    int C,
    int N,
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
    // Two-pass packing controls (one of each pair is null depending on the pass)
    const int32_t* __restrict__ block_accum, // [C * blocks_per_row] — null on pass 1, cumsum on pass 2
    int32_t* __restrict__ block_cnts,        // [C * blocks_per_row] — written on pass 1, null on pass 2
    // Pass 2 outputs (all NULL on pass 1)
    int32_t* __restrict__ indptr,       // [C + 1]  CSR row pointers
    int64_t* __restrict__ camera_ids,   // [nnz]    camera index for each visible pair
    int64_t* __restrict__ gaussian_ids, // [nnz]    Gaussian index for each visible pair
    float* __restrict__ means2d_out,    // [nnz, 2]
    float* __restrict__ conics_out,     // [nnz, 3]
    float* __restrict__ colors_out,     // [nnz, 3]
    float* __restrict__ opac_out,       // [nnz]
    float* __restrict__ depths_out,     // [nnz]
    int32_t* __restrict__ radii_out,    // [nnz, 2]
    // Per-Gaussian side outputs — always dense [N, ...], camera-independent
    // Written in Pass 1 by cam_idx == 0 threads only (avoid C redundant writes)
    float* __restrict__ means3d_out, // [N, 3]  conditioned 3D means
    float* __restrict__ wt_out       // [N]     temporal weights (used by densification)
)
{
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

        // Side outputs: written once per Gaussian during pass 1
        if (block_cnts != nullptr && cam_idx == 0) {
            means3d_out[gauss_idx * 3 + 0] = mx;
            means3d_out[gauss_idx * 3 + 1] = my;
            means3d_out[gauss_idx * 3 + 2] = mz;
            wt_out[gauss_idx] = wt;
        }
        // The projection's opacity cull, taken before any per-camera work: skips temporally gated Gaussians.
        valid = opacity_cond >= ALPHA_THRESHOLD;
    }

    // STAGE 2: World-to-Camera Transform

    float mc_x, mc_y, mc_z;
    float cc00, cc01, cc02, cc11, cc12, cc22;
    if (valid) {
        const float* vm = viewmats + cam_idx * 16;
        const float r00 = vm[0], r01 = vm[1], r02 = vm[2];
        const float r10 = vm[4], r11 = vm[5], r12 = vm[6];
        const float r20 = vm[8], r21 = vm[9], r22 = vm[10];
        const float tx = vm[3], ty = vm[7], tz = vm[11];

        mc_x = r00 * mx + r01 * my + r02 * mz + tx;
        mc_y = r10 * mx + r11 * my + r12 * mz + ty;
        mc_z = r20 * mx + r21 * my + r22 * mz + tz;

        float cov_3d_cond[6];
        device_gram_lower(cov_L, cov_3d_cond);
        device_covarW2C(r00, r01, r02, r10, r11, r12, r20, r21, r22, cov_3d_cond, cc00, cc01, cc02, cc11, cc12, cc22);
    }

    // STAGE 3: EWA Perspective Projection + Culling

    float mean2d_x, mean2d_y;
    float conic_a, conic_b, conic_c;
    float depth_out;
    int32_t radius_x, radius_y;
    float compensation = 1.0f;

    if (valid) {
        const float* K_ptr = Ks + cam_idx * 9;
        valid = device_ewa_project<F, kLens>(
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
            K_ptr[5],
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
    }

    // Two-pass packing

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        // --- Pass 1: count visible per block ---
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, N_THREADS_PACKED> BlockReduce;
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
        // --- Pass 2: scatter to packed positions ---
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, N_THREADS_PACKED> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if (valid) {
            if (block_idx > 0) {
                int32_t offset = block_accum[block_idx - 1];
                thread_data += offset;
            }
            // Write packed outputs
            camera_ids[thread_data] = cam_idx;
            gaussian_ids[thread_data] = gauss_idx;
            means2d_out[thread_data * 2 + 0] = mean2d_x;
            means2d_out[thread_data * 2 + 1] = mean2d_y;
            conics_out[thread_data * 3 + 0] = conic_a;
            conics_out[thread_data * 3 + 1] = conic_b;
            conics_out[thread_data * 3 + 2] = conic_c;
            // Colour only for visible pairs; pass 1 needs none.
            const float* vm = viewmats + cam_idx * 16;
            const float tx = vm[3], ty = vm[7], tz = vm[11];
            const float cam_x = -(vm[0] * tx + vm[4] * ty + vm[8] * tz);
            const float cam_y = -(vm[1] * tx + vm[5] * ty + vm[9] * tz);
            const float cam_z = -(vm[2] * tx + vm[6] * ty + vm[10] * tz);
            float r, g, b;
            AppearanceOps<M>::eval(
                sh_degree,
                n_temporal_terms,
                temporal_stride,
                mx - cam_x,
                my - cam_y,
                mz - cam_z,
                tau - mu_t[gauss_idx],
                ShCoeffs{sh0 + gauss_idx * 3, shN + gauss_idx * shN_stride},
                (sh_temporal != nullptr && n_temporal_terms > 0) ? sh_temporal + gauss_idx * temporal_stride : nullptr,
                r,
                g,
                b
            );
            colors_out[thread_data * 3 + 0] = r;
            colors_out[thread_data * 3 + 1] = g;
            colors_out[thread_data * 3 + 2] = b;
            opac_out[thread_data] = opacity_cond * compensation;
            depths_out[thread_data] = depth_out;
            radii_out[thread_data * 2 + 0] = radius_x;
            radii_out[thread_data * 2 + 1] = radius_y;
        }
        // Lane 0 of first block in each row writes indptr
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
 * Host entry: pass 1, cumsum, allocate, pass 2.
 *
 * Returns (camera_ids [nnz] int64, gaussian_ids [nnz] int64, indptr [C+1] int32, means2d [nnz, 2],
 * conics [nnz, 3], colors [nnz, 3], opac [nnz], depths [nnz], radii [nnz, 2] int32,
 * means_3d_cond [N, 3], wt [N]). An empty
 * sh_temporal disables temporal SH.
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel(
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

    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(N > 0 && C > 0, "N and C must be positive");

    auto opts_f = means.options();
    auto opts_i = means.options().dtype(c10::kInt);
    auto opts_l = means.options().dtype(c10::kLong);

    const int blocks_per_row = (N + N_THREADS_PACKED - 1) / N_THREADS_PACKED;
    const int total_blocks = C * blocks_per_row;
    dim3 threads(N_THREADS_PACKED);
    dim3 grid(blocks_per_row, C);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Allocate per-Gaussian side outputs (always dense)
    Tensor means3d_out = at::zeros({N, 3}, opts_f);
    Tensor wt_out = at::zeros({N}, opts_f);

    const float* sh_temporal_ptr = (sh_temporal.numel() > 0) ? sh_temporal.contiguous().data_ptr<float>() : nullptr;

    // --- Pass 1: count visible per block ---
    Tensor block_cnts = at::empty({total_blocks}, opts_i);

    ANYCAM4D_DISPATCH_EWA_LENS(
        lens_args.has_lens,
        family,
        ANYCAM4D_DISPATCH_APPEARANCE(
            appearance_kind,
            ewa_4d_fused_packed_conditioned_projection_fwd_kernel<F, kLens, M><<<grid, threads, 0, stream>>>(
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
                N,
                tau,
                cov_eps,
                eps2d,
                near_plane,
                far_plane,
                prefilter_threshold,
                image_width,
                image_height,
                calc_compensations,
                nullptr,                        // block_accum (pass 1)
                block_cnts.data_ptr<int32_t>(), // block_cnts
                nullptr,                        // indptr
                nullptr,                        // camera_ids
                nullptr,                        // gaussian_ids
                nullptr,                        // means2d
                nullptr,                        // conics
                nullptr,                        // colors
                nullptr,                        // opac
                nullptr,                        // depths
                nullptr,                        // radii
                means3d_out.data_ptr<float>(),
                wt_out.data_ptr<float>()
            )
        )
    );

    // Cumsum to get global offsets
    Tensor block_accum = at::cumsum(block_cnts, 0, c10::kInt);
    int32_t nnz = block_accum[-1].item<int32_t>();

    // Allocate packed output tensors
    Tensor camera_ids_out = at::empty({nnz}, opts_l);
    Tensor gaussian_ids_out = at::empty({nnz}, opts_l);
    Tensor indptr_out = at::empty({C + 1}, opts_i);
    Tensor means2d_out = at::empty({nnz, 2}, opts_f);
    Tensor conics_out = at::empty({nnz, 3}, opts_f);
    Tensor colors_out = at::empty({nnz, 3}, opts_f);
    Tensor opac_out = at::empty({nnz}, opts_f);
    Tensor depths_out = at::empty({nnz}, opts_f);
    Tensor radii_out = at::empty({nnz, 2}, opts_i);

    if (nnz > 0) {
        // --- Pass 2: scatter to packed positions ---
        ANYCAM4D_DISPATCH_EWA_LENS(
            lens_args.has_lens,
            family,
            ANYCAM4D_DISPATCH_APPEARANCE(
                appearance_kind,
                ewa_4d_fused_packed_conditioned_projection_fwd_kernel<F, kLens, M><<<grid, threads, 0, stream>>>(
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
                    N,
                    tau,
                    cov_eps,
                    eps2d,
                    near_plane,
                    far_plane,
                    prefilter_threshold,
                    image_width,
                    image_height,
                    calc_compensations,
                    block_accum.data_ptr<int32_t>(), // block_accum (pass 2)
                    nullptr,                         // block_cnts (pass 2)
                    indptr_out.data_ptr<int32_t>(),
                    camera_ids_out.data_ptr<int64_t>(),
                    gaussian_ids_out.data_ptr<int64_t>(),
                    means2d_out.data_ptr<float>(),
                    conics_out.data_ptr<float>(),
                    colors_out.data_ptr<float>(),
                    opac_out.data_ptr<float>(),
                    depths_out.data_ptr<float>(),
                    radii_out.data_ptr<int32_t>(),
                    nullptr, // means_3d_cond (already written in pass 1)
                    nullptr  // wt (already written in pass 1)
                )
            )
        );
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
        colors_out,
        opac_out,
        depths_out,
        radii_out,
        means3d_out,
        wt_out
    };
}
