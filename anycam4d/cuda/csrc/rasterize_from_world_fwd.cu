/**
 * World-space front-to-back compositing with the ray-Gaussian Mahalanobis distance on conditioned
 * 3D Gaussians. Tile structure mirrors gsplat's rasterize_to_pixels_3dgs_fwd_kernel: grid
 * (C, tile_h, tile_w), one thread per pixel, Gaussian batches loaded into shared memory.
 *
 * Outputs render_colors [C, H, W, D], render_alphas [C, H, W, 1] and last_ids [C, H, W] for the backward.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "rasterize_from_world.cuh"

using at::Tensor;

namespace cg = cooperative_groups;

static constexpr int TILE_SIZE = WORLD_RASTERIZE_TILE_SIZE;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────────────────────

template <ProjectionFamily F, uint32_t CDIM>
__launch_bounds__(TILE_SIZE* TILE_SIZE, 2) __global__ void rasterize_from_world_fwd_kernel(
    const uint32_t C,        // number of cameras
    const uint32_t N,        // number of Gaussians
    const uint32_t n_isects, // total number of tile intersections
    // Gaussian parameters (world space)
    const float* __restrict__ means_3d_cond, // [N, 3]
    const float* __restrict__ cov_factors,   // [N, 6] packed lower L, Σ = L Lᵀ
    const float* __restrict__ opacities,     // [N]
    const float* __restrict__ colors,        // [C, N, CDIM]
    // Camera view (POD, by value): viewmats/Ks/distortion + scalars
    CameraViewDevice cam,
    // Bitpacked observation mask [C, H, W_words] (nullptr ⇒ every pixel observed)
    const int32_t* __restrict__ valid_mask,
    const uint32_t mask_words,
    // Tile / intersection structure
    const int32_t* __restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t* __restrict__ flatten_ids,  // [n_isects]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // Outputs
    float* __restrict__ render_colors, // [C, H, W, CDIM]
    float* __restrict__ render_alphas, // [C, H, W]
    int32_t* __restrict__ last_ids     // [C, H, W]
)
{
    auto block = cg::this_thread_block();

    // Camera and tile indices from 3D grid (C, tile_height, tile_width)
    const uint32_t cam_id = block.group_index().x;
    const uint32_t tile_y = block.group_index().y;
    const uint32_t tile_x = block.group_index().z;
    const uint32_t tile_id = tile_y * tile_width + tile_x;

    // Pixel coordinates within the tile
    const uint32_t py = tile_y * TILE_SIZE + block.thread_index().y;
    const uint32_t px = tile_x * TILE_SIZE + block.thread_index().x;

    // Offset per-camera pointers to this camera
    tile_offsets += cam_id * tile_height * tile_width;
    render_colors += cam_id * image_height * image_width * CDIM;
    render_alphas += cam_id * image_height * image_width;
    last_ids += cam_id * image_height * image_width;

    const uint32_t pix_id = py * image_width + px;
    const bool inside = (py < image_height && px < image_width);

    // ── Camera ray generation ─────────────────────────────────────────────
    const float* vm = cam.viewmats + cam_id * 16;
    const float* K_p = cam.Ks + cam_id * 9;
    const float* dist = cam.distortion + cam_id * CameraOps<F>::n_slots;

    const int32_t* vmask = (valid_mask == nullptr) ? nullptr : valid_mask + cam_id * image_height * mask_words;

    // Shared fwd/bwd ray path — see device_pixel_ray (must stay bit-for-bit identical).
    float ox, oy, oz, dw_x, dw_y, dw_z;
    const bool ray_valid = device_pixel_ray<F>(vm, K_p, dist, (float)px, (float)py, ox, oy, oz, dw_x, dw_y, dw_z) &&
                           (!inside || device_pixel_observed(vmask, mask_words, px, py));

    // A pixel that was never observed, or whose ray does not invert through the camera model,
    // is never composited: the loop below is skipped entirely, so it keeps the zero-initialised
    // colour and T = 1 (alpha 0, no last contributor). Integrating a fabricated direction there
    // would write plausible-looking garbage that the loss then has to be told to ignore. The
    // gate must stay identical to the backward's — see rasterize_from_world_bwd.cu.
    bool done = !inside || !ray_valid;

    // ── Tile intersection range ───────────────────────────────────────────
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (cam_id == C - 1 && tile_id == tile_width * tile_height - 1) ? (int32_t)n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // ── Shared memory: one batch of Gaussians (μ, opacity, L), reused by every pixel of the tile.
    extern __shared__ __align__(16) int s[];
    const GaussianBatch batch = device_gaussian_batch(s, block_size);

    float T = 1.0f;
    // Use -1 as sentinel for "no Gaussian contributed".  If we left cur_idx = 0
    // and no Gaussian passed the alpha threshold, the backward kernel would
    // incorrectly process flatten_ids[0] for this pixel.
    int32_t cur_idx = -1;
    float pix_out[CDIM] = {};

    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size)
            break;

        // Cooperative load: each thread loads one Gaussian
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + block.thread_rank();
        if (idx < (uint32_t)range_end) {
            // flatten_ids stores cam_id*N + gauss_id; per-Gaussian tensors are indexed by gauss_id alone,
            // per-(cam, Gaussian) colours by the flat id.
            const int32_t g = flatten_ids[idx];
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
        for (uint32_t t = 0; t < batch_size && !done; ++t) {
            const int32_t g = batch.ids[t];
            float4 mo;
            float Lf[6];
            device_load_gaussian(batch, t, mo, Lf);
            const float opac = mo.w;

            // Mahalanobis squared distance from the ray to the Gaussian centre
            // (whitened by L from the ray's closest point, inverse-free)
            float a, d2, r[3];
            device_ray_gaussian_mahal(ox, oy, oz, dw_x, dw_y, dw_z, mo.x, mo.y, mo.z, Lf, a, r, d2);

            if (a <= 0.f)
                continue; // degenerate (a = ‖y_d‖²; only zero if d collapses)

            const float power = -0.5f * d2;

            const float max_response = __expf(power);
            const float alpha = fminf(MAX_RASTER_ALPHA, opac * max_response);
            // Dual cutoff (matches gsplat's RasterizeToPixelsFromWorld3DGSFwd):
            //   1. alpha < 1/255  → contribution negligible after quantisation.
            //   2. max_response   → ray is ≳ 3 σ from this Gaussian's centre.
            // The 2nd test is independent of opacity and prunes high-opacity
            // Gaussians whose tails would otherwise spread noise across the tile.
            if (alpha < ALPHA_THRESHOLD || max_response <= MAX_RASTER_KERNEL_DENSITY)
                continue;

            const float next_T = T * (1.f - alpha);
            if (next_T <= T_RASTER_THRESHOLD) {
                done = true;
                break;
            }

            const float vis = alpha * T;
            const float* c_ptr = colors + g * CDIM; // g = cam_id*N+gauss_id → flat index into [C,N,CDIM]
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k)
                pix_out[k] += c_ptr[k] * vis;

            cur_idx = (int32_t)(batch_start + t);
            T = next_T;
        }
    }

    if (inside) {
        render_alphas[pix_id] = 1.f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k)
            render_colors[pix_id * CDIM + k] = pix_out[k];
        last_ids[pix_id] = cur_idx;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Host entry. radial_coeffs is fixed-width OpenCV distortion (fisheye reads four slots); family is
 * 0 pinhole, 1 fisheye; valid_mask is the bitpacked observation mask or absent (dense path).
 * Returns (render_colors [C, H, W, D], render_alphas [C, H, W], last_ids [C, H, W]).
 */
std::tuple<Tensor, Tensor, Tensor> rasterize_from_world_fwd(
    const Tensor& means_3d_cond,
    const Tensor& cov_factors,
    const Tensor& opacities,
    const Tensor& colors,
    const Tensor& viewmats,
    const Tensor& Ks,
    const Tensor& radial_coeffs,
    const Tensor& tile_offsets,
    const Tensor& flatten_ids,
    int image_width,
    int image_height,
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
    auto opts_i = means_3d_cond.options().dtype(c10::kInt);
    Tensor render_colors = at::zeros({C, image_height, image_width, CDIM}, opts);
    Tensor render_alphas = at::zeros({C, image_height, image_width}, opts);
    Tensor last_ids = at::zeros({C, image_height, image_width}, opts_i);

    const dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);
    const dim3 grid_dim(C, tile_height, tile_width);

    // Shared memory: id[block] + means[block*3] + cov_factors[block*6] + opac[block]
    const int block_size = TILE_SIZE * TILE_SIZE;
    const size_t smem_bytes = GAUSSIAN_BATCH_BYTES * block_size;

    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_FWD(FAMILY, CDIM_VAL)                                                                                   \
    rasterize_from_world_fwd_kernel<FAMILY, CDIM_VAL><<<grid_dim, block_dim, smem_bytes, stream>>>(                    \
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
        render_colors.data_ptr<float>(),                                                                               \
        render_alphas.data_ptr<float>(),                                                                               \
        last_ids.data_ptr<int32_t>()                                                                                   \
    )

#define DISPATCH_CDIM_FWD(FAMILY)                                                                                      \
    do {                                                                                                               \
        if (CDIM == 3) {                                                                                               \
            LAUNCH_FWD(FAMILY, 3);                                                                                     \
        }                                                                                                              \
        else if (CDIM == 4) {                                                                                          \
            LAUNCH_FWD(FAMILY, 4);                                                                                     \
        }                                                                                                              \
        else if (CDIM == 1) {                                                                                          \
            LAUNCH_FWD(FAMILY, 1);                                                                                     \
        }                                                                                                              \
        else {                                                                                                         \
            AT_ERROR("rasterize_from_world_fwd: unsupported CDIM=", CDIM, ". Supported: 1, 3, 4.");                    \
        }                                                                                                              \
    } while (0)

    switch (family) {
    case PINHOLE_FAMILY:
        DISPATCH_CDIM_FWD(PINHOLE_FAMILY);
        break;
    case FISHEYE_FAMILY:
        DISPATCH_CDIM_FWD(FISHEYE_FAMILY);
        break;
    default:
        TORCH_CHECK(false, "unsupported projection family ", family);
    }
#undef DISPATCH_CDIM_FWD
#undef LAUNCH_FWD

    return {render_colors, render_alphas, last_ids};
}
