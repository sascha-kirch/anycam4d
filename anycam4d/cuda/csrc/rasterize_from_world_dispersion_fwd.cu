/**
 * Lateral chromatic aberration in one pass. The projection conditions and colours each Gaussian once and projects its
 * sigma points through every wavelength's lens, keeping the union of the boxes (invariant 4). The rasterizer casts 3·S
 * rays from the camera centre, ray r through its own lens (wavelength sample r of colour channel r / S), each composited
 * with its own transmittance into its channel. Gaussians, depth order and tile lists are shared; an unobserved pixel
 * (host mask) is never composited, as in the plain world rasterizer.
 * Ray r reproduces the plain world rasterizer through lens r, so S = 1 equals three single-lens renders. Forward only.
 *
 * Outputs render_colors [1, H, W, 3] (channel mean over its S rays) and render_alphas [1, H, W, 3] (likewise).
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "conditioning.cuh"
#include "rasterize_from_world.cuh"
#include "sh4d.cuh"
#include "ut_4d.cuh"

using at::Tensor;

namespace cg = cooperative_groups;

static constexpr int TILE_SIZE = WORLD_RASTERIZE_TILE_SIZE;

template <ProjectionFamily F, AppearanceKind M>
__launch_bounds__(256, 2) __global__ void ut_4d_dispersion_projection_kernel(
    const float* __restrict__ quats_l,
    const float* __restrict__ quats_r,
    const float* __restrict__ scales_4d_log,
    const float* __restrict__ means,
    const float* __restrict__ mu_t,
    const float* __restrict__ opacities_in,
    const float* __restrict__ sh0,
    const float* __restrict__ shN,
    const float* __restrict__ sh_temporal,
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    const float* __restrict__ viewmat, // [16]
    const float* __restrict__ Ks,      // [n_lenses, 9]
    const float* __restrict__ dist,    // [n_lenses, n_slots]
    int n_lenses,
    int image_width,
    int image_height,
    float tau,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    float* __restrict__ means2d_out,     // [N, 2] centre of the union box
    float* __restrict__ depths_out,      // [N]
    int32_t* __restrict__ radii_out,     // [N, 2] half-size of the union box
    float* __restrict__ colors_out,      // [N, 3]
    float* __restrict__ means3d_out,     // [N, 3]
    float* __restrict__ cov_factors_out, // [N, 6]
    float* __restrict__ opacities_out,   // [N]
    int N
)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N)
        return;

    float mx, my, mz, L[6], wt, opacity;
    device_condition_rot4d<float>(
        quats_l + g * 4,
        quats_r + g * 4,
        scales_4d_log + g * 4,
        means[g * 3 + 0],
        means[g * 3 + 1],
        means[g * 3 + 2],
        mu_t[g],
        opacities_in[g],
        tau,
        cov_eps,
        prefilter_threshold,
        mx,
        my,
        mz,
        L,
        wt,
        opacity
    );
    means3d_out[g * 3 + 0] = mx;
    means3d_out[g * 3 + 1] = my;
    means3d_out[g * 3 + 2] = mz;
#pragma unroll
    for (int i = 0; i < 6; ++i)
        cov_factors_out[g * 6 + i] = L[i];
    opacities_out[g] = opacity;

    // Union over the lenses of each lens's UT box, in the arithmetic of a box (mean ± radius) per lens.
    float lo_x = INFINITY, lo_y = INFINITY, hi_x = -INFINITY, hi_y = -INFINITY, depth = 0.f;
    if (opacity >= ALPHA_THRESHOLD) {
        float wm[7] = {UT_WM0, UT_WMI, UT_WMI, UT_WMI, UT_WMI, UT_WMI, UT_WMI};
        float wc[7] = {UT_WC0, UT_WCI, UT_WCI, UT_WCI, UT_WCI, UT_WCI, UT_WCI};
        const float mu[3] = {mx, my, mz};
        float pts[7][3];
        device_sigma_points_from_factor(mu, L, UT_C_SPREAD, pts);
        for (int k = 0; k < n_lenses; ++k) {
            const float* K = Ks + k * 9;
            float m2x, m2y, ca, cb, cc, d;
            int32_t rx, ry;
            const bool visible = device_ut_project<F>(
                pts,
                L,
                wm,
                wc,
                viewmat,
                K[0],
                K[4],
                K[2],
                K[5],
                eps2d,
                opacity,
                near_plane,
                far_plane,
                image_width,
                image_height,
                dist + k * CameraOps<F>::n_slots,
                m2x,
                m2y,
                ca,
                cb,
                cc,
                d,
                rx,
                ry
            );
            if (!visible || rx <= 0 || ry <= 0)
                continue;
            lo_x = fminf(lo_x, m2x - (float)rx);
            lo_y = fminf(lo_y, m2y - (float)ry);
            hi_x = fmaxf(hi_x, m2x + (float)rx);
            hi_y = fmaxf(hi_y, m2y + (float)ry);
            depth = d; // ‖μ_c‖: the same for every lens
        }
    }
    if (lo_x <= hi_x) {
        means2d_out[g * 2 + 0] = 0.5f * (lo_x + hi_x);
        means2d_out[g * 2 + 1] = 0.5f * (lo_y + hi_y);
        // +1 px: the float centre may sit an ulp off, and the box must still contain [lo, hi] (invariant 4)
        radii_out[g * 2 + 0] = (int32_t)ceilf(0.5f * (hi_x - lo_x)) + 1;
        radii_out[g * 2 + 1] = (int32_t)ceilf(0.5f * (hi_y - lo_y)) + 1;
        depths_out[g] = depth;
    }
    else {
        means2d_out[g * 2 + 0] = 0.f;
        means2d_out[g * 2 + 1] = 0.f;
        radii_out[g * 2 + 0] = 0;
        radii_out[g * 2 + 1] = 0;
        depths_out[g] = 0.f;
    }

    // Colour from the camera centre, which every lens shares.
    const float tx = viewmat[3], ty = viewmat[7], tz = viewmat[11];
    const float cam_x = -(viewmat[0] * tx + viewmat[4] * ty + viewmat[8] * tz);
    const float cam_y = -(viewmat[1] * tx + viewmat[5] * ty + viewmat[9] * tz);
    const float cam_z = -(viewmat[2] * tx + viewmat[6] * ty + viewmat[10] * tz);
    float r, gr, b;
    AppearanceOps<M>::eval(
        sh_degree,
        n_temporal_terms,
        temporal_stride,
        mx - cam_x,
        my - cam_y,
        mz - cam_z,
        tau - mu_t[g],
        ShCoeffs{sh0 + g * 3, shN + g * shN_stride},
        (sh_temporal != nullptr && n_temporal_terms > 0) ? sh_temporal + g * temporal_stride : nullptr,
        r,
        gr,
        b
    );
    colors_out[g * 3 + 0] = r;
    colors_out[g * 3 + 1] = gr;
    colors_out[g * 3 + 2] = b;
}

/** Returns (means2d [1,N,2], depths [1,N], radii [1,N,2], colors [1,N,3], means [N,3], cov_factors [N,6], opacities [N]). */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> ut_4d_dispersion_projection_fwd(
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
    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(0) == 1, "dispersion renders one pose: viewmats [1, 4, 4]");
    const int n_lenses = (int)Ks.size(0);
    const int slots = (family == FISHEYE_FAMILY) ? 4 : 12;
    TORCH_CHECK(Ks.dim() == 3 && n_lenses >= 1, "Ks must be [L, 3, 3]");
    TORCH_CHECK(
        radial_coeffs.dim() == 2 && radial_coeffs.size(0) == n_lenses && radial_coeffs.size(1) == slots,
        "radial_coeffs must be [L, ",
        slots,
        "]"
    );
    TORCH_CHECK(
        viewmats.is_contiguous() && Ks.is_contiguous() && radial_coeffs.is_contiguous(),
        "camera tensors must be contiguous"
    );
    const int N = (int)means.size(0);
    auto opts = means.options();
    Tensor means2d = at::zeros({1, N, 2}, opts);
    Tensor depths = at::zeros({1, N}, opts);
    Tensor radii = at::zeros({1, N, 2}, opts.dtype(c10::kInt));
    Tensor colors = at::zeros({1, N, 3}, opts);
    Tensor means3d = at::zeros({N, 3}, opts);
    Tensor cov_factors = at::zeros({N, 6}, opts);
    Tensor opac = at::zeros({N}, opts);
    if (N == 0)
        return {means2d, depths, radii, colors, means3d, cov_factors, opac};

    const Tensor sh_temporal_c = sh_temporal.contiguous();
    const float* sh_temporal_ptr = (sh_temporal_c.numel() > 0) ? sh_temporal_c.data_ptr<float>() : nullptr;
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_DISPERSION_PROJ(FAMILY)                                                                                 \
    ut_4d_dispersion_projection_kernel<FAMILY, M><<<grid_size, block_size, 0, stream>>>(                               \
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
        viewmats.data_ptr<float>(),                                                                                    \
        Ks.data_ptr<float>(),                                                                                          \
        radial_coeffs.data_ptr<float>(),                                                                               \
        n_lenses,                                                                                                      \
        image_width,                                                                                                   \
        image_height,                                                                                                  \
        tau,                                                                                                           \
        cov_eps,                                                                                                       \
        eps2d,                                                                                                         \
        near_plane,                                                                                                    \
        far_plane,                                                                                                     \
        prefilter_threshold,                                                                                           \
        means2d.data_ptr<float>(),                                                                                     \
        depths.data_ptr<float>(),                                                                                      \
        radii.data_ptr<int32_t>(),                                                                                     \
        colors.data_ptr<float>(),                                                                                      \
        means3d.data_ptr<float>(),                                                                                     \
        cov_factors.data_ptr<float>(),                                                                                 \
        opac.data_ptr<float>(),                                                                                        \
        N                                                                                                              \
    )

    switch (family) {
    case PINHOLE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_DISPERSION_PROJ(PINHOLE_FAMILY));
        break;
    case FISHEYE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_DISPERSION_PROJ(FISHEYE_FAMILY));
        break;
    default:
        TORCH_CHECK(false, "unsupported projection family ", family);
    }
#undef LAUNCH_DISPERSION_PROJ
    return {means2d, depths, radii, colors, means3d, cov_factors, opac};
}

template <ProjectionFamily F, uint32_t S>
__launch_bounds__(TILE_SIZE* TILE_SIZE, 2) __global__ void rasterize_from_world_dispersion_fwd_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    const float* __restrict__ means_3d_cond, // [N, 3]
    const float* __restrict__ cov_factors,   // [N, 6]
    const float* __restrict__ opacities,     // [N]
    const float* __restrict__ colors,        // [N, 3]
    const float* __restrict__ viewmat,       // [16], shared by every lens
    const float* __restrict__ Ks,            // [3S, 9]
    const float* __restrict__ dist,          // [3S, n_slots]
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const int32_t* __restrict__ valid_mask, // [H, W_words] bitpacked, nullptr ⇒ every pixel observed
    const uint32_t mask_words,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    float* __restrict__ render_colors, // [H, W, 3]
    float* __restrict__ render_alphas  // [H, W, 3]
)
{
    constexpr uint32_t NR = 3 * S;
    auto block = cg::this_thread_block();
    const uint32_t tile_y = block.group_index().y;
    const uint32_t tile_x = block.group_index().z;
    const uint32_t tile_id = tile_y * tile_width + tile_x;
    const uint32_t py = tile_y * TILE_SIZE + block.thread_index().y;
    const uint32_t px = tile_x * TILE_SIZE + block.thread_index().x;
    const uint32_t pix_id = py * image_width + px;
    const bool inside = (py < image_height && px < image_width);
    const bool observed = inside && device_pixel_observed(valid_mask, mask_words, px, py);

    // One origin, NR directions; a ray outside its lens's invertible domain is never composited.
    float ox, oy, oz, dir[NR][3], T[NR];
    bool alive[NR];
    bool done = true;
#pragma unroll
    for (uint32_t r = 0; r < NR; ++r) {
        alive[r] = device_pixel_ray<F>(
                       viewmat,
                       Ks + r * 9,
                       dist + r * CameraOps<F>::n_slots,
                       (float)px,
                       (float)py,
                       ox,
                       oy,
                       oz,
                       dir[r][0],
                       dir[r][1],
                       dir[r][2]
                   ) &&
                   observed;
        T[r] = 1.f;
        done = done && !alive[r];
    }

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (tile_id == tile_width * tile_height - 1) ? (int32_t)n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ __align__(16) int s[];
    const GaussianBatch batch = device_gaussian_batch(s, block_size);

    float pix_out[3] = {};
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size)
            break;
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + block.thread_rank();
        if (idx < (uint32_t)range_end) {
            const int32_t g = flatten_ids[idx];
            device_store_gaussian(batch, block.thread_rank(), g, g, means_3d_cond, cov_factors, opacities);
        }
        block.sync();

        const uint32_t batch_size = min(block_size, (uint32_t)(range_end - (int32_t)batch_start));
        for (uint32_t t = 0; t < batch_size && !done; ++t) {
            float4 mo;
            float Lf[6];
            device_load_gaussian(batch, t, mo, Lf);
            const float* c_ptr = colors + batch.ids[t] * 3;
            done = true;
#pragma unroll
            for (uint32_t r = 0; r < NR; ++r) {
                if (!alive[r])
                    continue;
                done = false;
                float a, d2, res[3];
                device_ray_gaussian_mahal(ox, oy, oz, dir[r][0], dir[r][1], dir[r][2], mo.x, mo.y, mo.z, Lf, a, res, d2);
                if (a <= 0.f)
                    continue;
                const float max_response = __expf(-0.5f * d2);
                const float alpha = fminf(MAX_RASTER_ALPHA, mo.w * max_response);
                if (alpha < ALPHA_THRESHOLD || max_response <= MAX_RASTER_KERNEL_DENSITY)
                    continue;
                const float next_T = T[r] * (1.f - alpha);
                if (next_T <= T_RASTER_THRESHOLD) {
                    alive[r] = false; // as the single-lens kernel: stop before this Gaussian
                    continue;
                }
                pix_out[r / S] += c_ptr[r / S] * alpha * T[r];
                T[r] = next_T;
            }
            if (!done) {
                done = true;
#pragma unroll
                for (uint32_t r = 0; r < NR; ++r)
                    done = done && !alive[r];
            }
        }
    }

    if (inside) {
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            float t_sum = 0.f;
#pragma unroll
            for (uint32_t k = 0; k < S; ++k)
                t_sum += T[c * S + k];
            render_colors[pix_id * 3 + c] = pix_out[c] / (float)S;
            render_alphas[pix_id * 3 + c] = 1.f - t_sum / (float)S;
        }
    }
}

/** Returns (render_colors [1, H, W, 3], render_alphas [1, H, W, 3]). Ks [3S, 3, 3], radial_coeffs [3S, P]; valid_mask
 * [1, H, (W+31)/32] or absent. */
std::tuple<Tensor, Tensor> rasterize_from_world_dispersion_fwd(
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
    const int N = (int)means_3d_cond.size(0);
    const int n_rays = (int)Ks.size(0);
    for (const Tensor* t : {&means_3d_cond, &cov_factors, &opacities, &colors, &viewmats, &Ks, &radial_coeffs})
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat, "dispersion inputs must be float32 CUDA tensors");
    TORCH_CHECK(
        tile_offsets.is_cuda() && tile_offsets.scalar_type() == at::kInt && flatten_ids.scalar_type() == at::kInt,
        "tile_offsets and flatten_ids must be int32 CUDA tensors"
    );
    const int samples = n_rays / 3;
    const int slots = (family == FISHEYE_FAMILY) ? 4 : 12;
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(0) == 1, "dispersion renders one pose: viewmats [1, 4, 4]");
    TORCH_CHECK(
        n_rays % 3 == 0 && (samples == 1 || samples == 2 || samples == 4 || samples == 8),
        "Ks must be [3S, 3, 3] with S in {1, 2, 4, 8}; got ",
        n_rays,
        " lenses"
    );
    TORCH_CHECK(
        radial_coeffs.dim() == 2 && radial_coeffs.size(0) == n_rays && radial_coeffs.size(1) == slots,
        "radial_coeffs must be [3S, ",
        slots,
        "]"
    );
    TORCH_CHECK(
        colors.dim() == 3 && colors.size(0) == 1 && colors.size(1) == N && colors.size(2) == 3, "colors must be [1, N, 3]"
    );
    TORCH_CHECK(
        tile_offsets.dim() == 3 && tile_offsets.size(0) == 1 &&
            tile_offsets.size(1) == (image_height + TILE_SIZE - 1) / TILE_SIZE &&
            tile_offsets.size(2) == (image_width + TILE_SIZE - 1) / TILE_SIZE,
        "tile_offsets must be [1, ceil(H / tile), ceil(W / tile)]"
    );
    const auto [vmask_ptr, mask_words] = device_valid_mask_ptr(valid_mask, 1, image_width, image_height);
    TORCH_CHECK(
        viewmats.is_contiguous() && Ks.is_contiguous() && radial_coeffs.is_contiguous(),
        "camera tensors must be contiguous"
    );

    const int tile_height = (int)tile_offsets.size(1);
    const int tile_width = (int)tile_offsets.size(2);
    auto opts = means_3d_cond.options();
    Tensor render_colors = at::zeros({1, image_height, image_width, 3}, opts);
    Tensor render_alphas = at::zeros({1, image_height, image_width, 3}, opts);

    const Tensor means_c = means_3d_cond.contiguous(), factors_c = cov_factors.contiguous();
    const Tensor opac_c = opacities.contiguous(), colors_c = colors.contiguous();
    const Tensor offsets_c = tile_offsets.contiguous(), ids_c = flatten_ids.contiguous();
    const dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);
    const dim3 grid_dim(1, tile_height, tile_width);
    const size_t smem_bytes = GAUSSIAN_BATCH_BYTES * TILE_SIZE * TILE_SIZE;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_DISPERSION(FAMILY, S_VAL)                                                                               \
    rasterize_from_world_dispersion_fwd_kernel<FAMILY, S_VAL><<<grid_dim, block_dim, smem_bytes, stream>>>(           \
        N,                                                                                                             \
        (uint32_t)flatten_ids.size(0),                                                                                 \
        means_c.data_ptr<float>(),                                                                                     \
        factors_c.data_ptr<float>(),                                                                                   \
        opac_c.data_ptr<float>(),                                                                                      \
        colors_c.data_ptr<float>(),                                                                                    \
        viewmats.data_ptr<float>(),                                                                                    \
        Ks.data_ptr<float>(),                                                                                          \
        radial_coeffs.data_ptr<float>(),                                                                               \
        offsets_c.data_ptr<int32_t>(),                                                                                 \
        ids_c.data_ptr<int32_t>(),                                                                                     \
        vmask_ptr,                                                                                                     \
        mask_words,                                                                                                    \
        image_width,                                                                                                   \
        image_height,                                                                                                  \
        tile_width,                                                                                                    \
        tile_height,                                                                                                   \
        render_colors.data_ptr<float>(),                                                                               \
        render_alphas.data_ptr<float>()                                                                                \
    )

#define DISPATCH_SAMPLES(FAMILY)                                                                                       \
    switch (samples) {                                                                                                 \
    case 1:                                                                                                            \
        LAUNCH_DISPERSION(FAMILY, 1);                                                                                  \
        break;                                                                                                         \
    case 2:                                                                                                            \
        LAUNCH_DISPERSION(FAMILY, 2);                                                                                  \
        break;                                                                                                         \
    case 4:                                                                                                            \
        LAUNCH_DISPERSION(FAMILY, 4);                                                                                  \
        break;                                                                                                         \
    default:                                                                                                           \
        LAUNCH_DISPERSION(FAMILY, 8);                                                                                  \
    }

    if (N > 0 && flatten_ids.size(0) > 0) {
        switch (family) {
        case PINHOLE_FAMILY:
            DISPATCH_SAMPLES(PINHOLE_FAMILY);
            break;
        case FISHEYE_FAMILY:
            DISPATCH_SAMPLES(FISHEYE_FAMILY);
            break;
        default:
            TORCH_CHECK(false, "unsupported projection family ", family);
        }
    }
#undef DISPATCH_SAMPLES
#undef LAUNCH_DISPERSION
    return {render_colors, render_alphas};
}
