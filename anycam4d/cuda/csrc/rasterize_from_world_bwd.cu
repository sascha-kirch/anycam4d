/**
 * World-space rasterization backward, walking each pixel's intersections back from last_ids
 * (gsplat RasterizeToPixels3DGSBwd).
 *
 * Forward: T_0 = 1, T_{i+1} = T_i (1 - α_i), C = Σ T_i α_i c_i. Backward, per Gaussian (i decreasing),
 * with accum = Σ_{j>i} T_j α_j c_j and T = T_{i+1}:
 *   remaining = accum / T,  T_i = T / (1 - α_i)
 *   ∂L/∂α_i = Σ_c dL/dC[c] · T_i · (c_i[c] - remaining[c]) - dL/dA · T_i
 *   ∂L/∂c_i = dL/dC · T_i · α_i
 *   accum += T_i α_i c_i,  T = T_i
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "rasterize_from_world.cuh"

using at::Tensor;

namespace cg = cooperative_groups;

static constexpr int TILE_SIZE_BWD = WORLD_RASTERIZE_TILE_SIZE;

constexpr int next_pow2(int x)
{
    int p = 1;
    while (p < x)
        p <<= 1;
    return p;
}

/**
 * Warp sum of V values (V a power of two <= 32) by recursive halving: at each level a lane keeps N/2 of its N values
 * and adds its partner's copy of that half, so V - 1 + (5 - log2 V) shuffles replace 5·V. On return, every lane l
 * holds the full sum of value l >> (5 - log2 V) in v[0]. Levels recurse at compile time so every index is constant.
 */
template <int N, int OFFSET, int V>
__device__ __forceinline__ void device_warp_reduce_scatter(float (&v)[V], uint32_t lane)
{
    if constexpr (N > 1) {
        constexpr int half = N / 2;
        const bool upper = (lane & OFFSET) != 0u;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float send = upper ? v[i] : v[i + half];
            const float keep = upper ? v[i + half] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, OFFSET);
        }
        device_warp_reduce_scatter<half, OFFSET / 2>(v, lane);
    }
    else if constexpr (OFFSET > 0) {
        v[0] += __shfl_xor_sync(0xffffffffu, v[0], OFFSET);
        device_warp_reduce_scatter<1, OFFSET / 2>(v, lane);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward kernel
// ─────────────────────────────────────────────────────────────────────────────

template <ProjectionFamily F, uint32_t CDIM>
__launch_bounds__(TILE_SIZE_BWD* TILE_SIZE_BWD, 2) __global__ void rasterize_from_world_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // Gaussian parameters (same as forward)
    const float* __restrict__ means_3d_cond, // [N, 3]
    const float* __restrict__ cov_factors,   // [N, 6] packed lower L
    const float* __restrict__ opacities,     // [N]
    const float* __restrict__ colors,        // [C, N, CDIM]
    // Camera view (POD, by value)
    CameraViewDevice cam,
    // Bitpacked observation mask [C, H, W_words] (nullptr ⇒ every pixel observed)
    const int32_t* __restrict__ valid_mask,
    const uint32_t mask_words,
    // Tile structure
    const int32_t* __restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t* __restrict__ flatten_ids,  // [n_isects]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // Forward outputs
    const float* __restrict__ render_alphas, // [C, H, W]
    const int32_t* __restrict__ last_ids,    // [C, H, W]
    // Upstream gradients
    const float* __restrict__ dL_drender_colors, // [C, H, W, CDIM]
    const float* __restrict__ dL_drender_alphas, // [C, H, W]
    // Gradient outputs
    float* __restrict__ dL_dmeans3d,     // [N, 3]
    float* __restrict__ dL_dcov_factors, // [N, 6]
    float* __restrict__ dL_dopacities,   // [N]
    float* __restrict__ dL_dcolors       // [C, N, CDIM]
)
{
    auto block = cg::this_thread_block();
    const cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const uint32_t cam_id = block.group_index().x;
    const uint32_t tile_y = block.group_index().y;
    const uint32_t tile_x = block.group_index().z;
    const uint32_t tile_id = tile_y * tile_width + tile_x;

    const uint32_t py = tile_y * TILE_SIZE_BWD + block.thread_index().y;
    const uint32_t px = tile_x * TILE_SIZE_BWD + block.thread_index().x;

    tile_offsets += cam_id * tile_height * tile_width;
    render_alphas += cam_id * image_height * image_width;
    last_ids += cam_id * image_height * image_width;
    dL_drender_colors += cam_id * image_height * image_width * CDIM;
    dL_drender_alphas += cam_id * image_height * image_width;

    const uint32_t pix_id = py * image_width + px;
    const bool inside = (py < image_height && px < image_width);

    // ── Per-pixel upstream gradients and final transmittance ─────────────
    float dL_dout[CDIM] = {};
    float dL_da_pixel = 0.f;
    float T_final_val = 1.f; // T_K = 1 − render_alpha, constant throughout backward
    float T = 1.f;           // T_{after_i}: decreases as we move forward (increases going bwd)
    int32_t last_idx = -1;

    if (inside) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k)
            dL_dout[k] = dL_drender_colors[pix_id * CDIM + k];
        dL_da_pixel = dL_drender_alphas[pix_id];
        T_final_val = 1.f - render_alphas[pix_id];
        T = T_final_val;
        last_idx = last_ids[pix_id];
    }

    // ── Camera ray ────────────────────────────────────────────────────────
    const float* vm = cam.viewmats + cam_id * 16;
    const float* K_p = cam.Ks + cam_id * 9;
    const float* dist = cam.distortion + cam_id * CameraOps<F>::n_slots;

    const int32_t* vmask = (valid_mask == nullptr) ? nullptr : valid_mask + cam_id * image_height * mask_words;

    // Shared fwd/bwd ray path — see device_pixel_ray (must stay bit-for-bit identical
    // to the forward; the backward differentiates the forward's ray).
    float ox, oy, oz, dw_x, dw_y, dw_z;
    const bool ray_valid = device_pixel_ray<F>(vm, K_p, dist, (float)px, (float)py, ox, oy, oz, dw_x, dw_y, dw_z) &&
                           (!inside || device_pixel_observed(vmask, mask_words, px, py));

    // Same gate as the forward: an unobserved pixel, or one whose ray does not invert,
    // composited nothing, so it has no reverse pass to run and must scatter no gradient. (The
    // forward left it with last_ids = -1, which already skips the loop; stating the condition
    // explicitly keeps the two kernels' gates textually identical and skips the work up front.)
    const bool contributes = inside && ray_valid;

    // ── Tile range ────────────────────────────────────────────────────────
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (cam_id == C - 1 && tile_id == tile_width * tile_height - 1) ? (int32_t)n_isects : tile_offsets[tile_id + 1];

    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // ── Shared memory: one batch of Gaussians, factored exactly as the forward does ──
    extern __shared__ __align__(16) int sb[];
    const GaussianBatch batch = device_gaussian_batch(sb, block_size);

    // Colour accumulated from Gaussians AFTER the current one (going backwards)
    float accum[CDIM] = {};

    // ── Reverse batch loop ────────────────────────────────────────────────
    for (int32_t b = (int32_t)num_batches - 1; b >= 0; --b) {
        const uint32_t batch_start = (uint32_t)(range_start + block_size * b);
        const uint32_t idx = batch_start + block.thread_rank();

        if (idx < (uint32_t)range_end) {
            const int32_t g = flatten_ids[idx]; // flat: cam_id*N + gauss_id
            device_store_gaussian(
                batch,
                block.thread_rank(),
                g,
                (int32_t)((uint32_t)g % N),
                means_3d_cond,
                cov_factors,
                opacities
            );
        }
        block.sync();

        const uint32_t batch_size = min(block_size, (uint32_t)(range_end - (int32_t)batch_start));

        // Process batch in reverse order (high global index first)
        for (int32_t t = (int32_t)batch_size - 1; t >= 0; --t) {
            const uint32_t global_t = batch_start + (uint32_t)t;

            // Lanes whose pixel the forward did not composite this Gaussian for contribute zeros; every lane joins
            // the reduction, so the warp skips only when no lane is valid.
            bool valid = contributes && (int32_t)global_t <= last_idx;
            float a = 0.f, d2 = 0.f, r[3] = {0.f, 0.f, 0.f}, alpha = 0.f;
            float4 mo;
            float Lf[6];
            device_load_gaussian(batch, (uint32_t)t, mo, Lf);
            const float opac = mo.w;
            if (valid) {
                device_ray_gaussian_mahal(ox, oy, oz, dw_x, dw_y, dw_z, mo.x, mo.y, mo.z, Lf, a, r, d2);
                const float max_response = __expf(-0.5f * d2);
                alpha = fminf(MAX_RASTER_ALPHA, opac * max_response);
                // Mirror the forward's cutoffs exactly, or bwd scatters gradient for what fwd skipped.
                valid = a > 0.f && alpha >= ALPHA_THRESHOLD && max_response > MAX_RASTER_KERNEL_DENSITY;
            }
            if (!warp.any(valid))
                continue;

            // g = flat cam_id*N+gauss_id; colors/dL_dcolors are [C,N,CDIM] → flat index correct
            const int32_t g = batch.ids[t];
            const int32_t gauss_id = (int32_t)((uint32_t)g % N);
            const float* c_ptr = colors + g * CDIM;

            float dL_dcolor[CDIM] = {};
            float dL_dmu[3] = {0.f, 0.f, 0.f}, dL_dfactor[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, dL_dopacity_i = 0.f;
            if (valid) {
                // T_i = T_{i+1} / (1 − α_i)
                const float one_minus_alpha = 1.f - alpha;
                const float T_before = (fabsf(one_minus_alpha) > 1e-6f) ? T / one_minus_alpha : T;
                float dL_dalpha_i = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    dL_dcolor[k] = T_before * alpha * dL_dout[k];
                    // remaining_{i+1}[c] = accum[c] / T  (T = T_{i+1} at this point)
                    const float remaining = (T > 1e-8f) ? (accum[k] / T) : 0.f;
                    dL_dalpha_i += dL_dout[k] * T_before * (c_ptr[k] - remaining);
                }
                // ∂(1 − T_K)/∂α_i = T_K / (1 − α_i) = T_final_val * T_before / T
                const float t_ratio = (T > 1e-8f) ? (T_final_val / T) : 0.f;
                dL_dalpha_i += dL_da_pixel * T_before * t_ratio;
                device_mahal_backward(dL_dalpha_i, opac, d2, r, Lf, dL_dmu, dL_dfactor, dL_dopacity_i);

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k)
                    accum[k] += T_before * alpha * c_ptr[k];
                T = T_before;
            }

            // One reduce-scatter over [color | μ | L | opacity]; lane group idx then owns gradient idx.
            constexpr int NV = CDIM + 10;
            constexpr int VP = next_pow2(NV);
            static_assert(VP <= 32, "one warp reduces at most 32 gradient values");
            float vals[VP] = {};
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k)
                vals[k] = dL_dcolor[k];
#pragma unroll
            for (int k = 0; k < 3; ++k)
                vals[CDIM + k] = dL_dmu[k];
#pragma unroll
            for (int k = 0; k < 6; ++k)
                vals[CDIM + 3 + k] = dL_dfactor[k];
            vals[CDIM + 9] = dL_dopacity_i;
            const uint32_t lane = warp.thread_rank();
            device_warp_reduce_scatter<VP, 16>(vals, lane);
            const float sum = vals[0];
            constexpr uint32_t group = 32 / VP;
            const uint32_t grad_slot = lane / group;
            if (lane % group == 0 && grad_slot < (uint32_t)NV) {
                float* dst;
                if (grad_slot < CDIM)
                    dst = dL_dcolors + g * CDIM + grad_slot;
                else if (grad_slot < CDIM + 3)
                    dst = dL_dmeans3d + gauss_id * 3 + (grad_slot - CDIM);
                else if (grad_slot < CDIM + 9)
                    dst = dL_dcov_factors + gauss_id * 6 + (grad_slot - CDIM - 3);
                else
                    dst = dL_dopacities + gauss_id;
                atomicAdd(dst, sum);
            }
        }
        // The next batch overwrites the shared slots other warps may still be reading.
        block.sync();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Host entry. valid_mask must be the one the forward used, or the gates disagree.
 * Returns (dL_dmeans3d [N, 3], dL_dcov_factors [N, 6], dL_dopacities [N], dL_dcolors [C, N, D]).
 */
std::tuple<Tensor, Tensor, Tensor, Tensor> rasterize_from_world_bwd(
    const Tensor& means_3d_cond,
    const Tensor& cov_factors,
    const Tensor& opacities,
    const Tensor& colors,
    const Tensor& viewmats,
    const Tensor& Ks,
    const Tensor& radial_coeffs,
    const Tensor& tile_offsets,
    const Tensor& flatten_ids,
    const Tensor& render_alphas,
    const Tensor& last_ids,
    const Tensor& dL_drender_colors,
    const Tensor& dL_drender_alphas,
    int family,
    const std::optional<Tensor>& valid_mask
)
{
    const int C = (int)viewmats.size(0);
    const int N = (int)means_3d_cond.size(0);
    const int CDIM = (int)colors.size(2);
    const int n_isects = (int)flatten_ids.size(0);
    const int tile_height = (int)tile_offsets.size(1);
    const int tile_width = (int)tile_offsets.size(2);
    const int image_height = (int)render_alphas.size(1);
    const int image_width = (int)render_alphas.size(2);

    // Defense-in-depth: fixed-width [C, P] with the family slot count (4 fisheye /
    // 12 pinhole-family). The host switch's exhaustive default is the only model gate.
    const int expected_stride = (family == FISHEYE_FAMILY) ? 4 : 12;
    TORCH_CHECK(
        radial_coeffs.dim() == 2 && radial_coeffs.size(0) == C && radial_coeffs.size(1) == expected_stride,
        "radial_coeffs must be [C, ",
        expected_stride,
        "] for this projection family; got a tensor of dim ",
        radial_coeffs.dim()
    );
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

    const auto [vmask_ptr, mask_words] = device_valid_mask_ptr(valid_mask, C, image_width, image_height);

    auto opts = means_3d_cond.options();
    Tensor dL_dmeans3d = at::zeros({N, 3}, opts);
    Tensor dL_dcov_factors = at::zeros({N, 6}, opts);
    Tensor dL_dopacities = at::zeros({N}, opts);
    Tensor dL_dcolors = at::zeros({C, N, CDIM}, opts);

    const dim3 block_dim(TILE_SIZE_BWD, TILE_SIZE_BWD, 1);
    const dim3 grid_dim(C, tile_height, tile_width);

    const int block_size = TILE_SIZE_BWD * TILE_SIZE_BWD;
    const size_t smem_bytes = GAUSSIAN_BATCH_BYTES * block_size;

    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_BWD(FAMILY, CDIM_VAL)                                                                                   \
    rasterize_from_world_bwd_kernel<FAMILY, CDIM_VAL><<<grid_dim, block_dim, smem_bytes, stream>>>(                    \
        C,                                                                                                             \
        N,                                                                                                             \
        n_isects,                                                                                                      \
        means_3d_cond.contiguous().data_ptr<float>(),                                                                  \
        cov_factors.contiguous().data_ptr<float>(),                                                                    \
        opacities.contiguous().data_ptr<float>(),                                                                      \
        colors.contiguous().data_ptr<float>(),                                                                         \
        cam,                                                                                                           \
        vmask_ptr,                                                                                                     \
        mask_words,                                                                                                    \
        tile_offsets.contiguous().data_ptr<int32_t>(),                                                                 \
        flatten_ids.contiguous().data_ptr<int32_t>(),                                                                  \
        image_width,                                                                                                   \
        image_height,                                                                                                  \
        tile_width,                                                                                                    \
        tile_height,                                                                                                   \
        render_alphas.contiguous().data_ptr<float>(),                                                                  \
        last_ids.contiguous().data_ptr<int32_t>(),                                                                     \
        dL_drender_colors.contiguous().data_ptr<float>(),                                                              \
        dL_drender_alphas.contiguous().data_ptr<float>(),                                                              \
        dL_dmeans3d.data_ptr<float>(),                                                                                 \
        dL_dcov_factors.data_ptr<float>(),                                                                             \
        dL_dopacities.data_ptr<float>(),                                                                               \
        dL_dcolors.data_ptr<float>()                                                                                   \
    )

#define DISPATCH_CDIM_BWD(FAMILY)                                                                                      \
    do {                                                                                                               \
        if (CDIM == 3) {                                                                                               \
            LAUNCH_BWD(FAMILY, 3);                                                                                     \
        }                                                                                                              \
        else if (CDIM == 4) {                                                                                          \
            LAUNCH_BWD(FAMILY, 4);                                                                                     \
        }                                                                                                              \
        else if (CDIM == 1) {                                                                                          \
            LAUNCH_BWD(FAMILY, 1);                                                                                     \
        }                                                                                                              \
        else {                                                                                                         \
            AT_ERROR("rasterize_from_world_bwd: unsupported CDIM=", CDIM, ". Supported: 1, 3, 4.");                    \
        }                                                                                                              \
    } while (0)

    switch (family) {
    case PINHOLE_FAMILY:
        DISPATCH_CDIM_BWD(PINHOLE_FAMILY);
        break;
    case FISHEYE_FAMILY:
        DISPATCH_CDIM_BWD(FISHEYE_FAMILY);
        break;
    default:
        TORCH_CHECK(false, "unsupported projection family ", family);
    }
#undef DISPATCH_CDIM_BWD
#undef LAUNCH_BWD

    return {dL_dmeans3d, dL_dcov_factors, dL_dopacities, dL_dcolors};
}
