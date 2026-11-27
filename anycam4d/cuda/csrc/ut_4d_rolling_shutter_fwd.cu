/**
 * Rolling-shutter forward render of 4D Gaussians (UT, world-space rasterization; no backward).
 *
 * Row r of camera c is exposed at τ_r = tau_first + r · tau_step through its own viewmat row_viewmats[c, r]. The
 * covariance is τ-independent, so each Gaussian is conditioned once at tau_ref; a row then needs only
 * mean + v (τ_r − tau_ref) and the gate at τ_r. Row r equals row r of the global-shutter render at (τ_r, viewmat_r):
 * a Gaussian is drawn on row r only in the tiles the global-shutter projection at that row assigns it, and the tile
 * box is the union of those (row, tile) pairs. Blend order is one depth per Gaussian; colour is evaluated once, at
 * tau_ref from the camera centre midway between the two middle rows.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>
#include <cfloat>
#include <climits>
#include <cooperative_groups.h>

#include "conditioning.cuh"
#include "rasterize_from_world.cuh"
#include "sh4d.cuh"
#include "ut_4d.cuh"

using at::Tensor;

namespace cg = cooperative_groups;

static constexpr int TILE_SIZE = WORLD_RASTERIZE_TILE_SIZE;
static_assert(TILE_SIZE <= 32, "one admission bit per tile row in a uint32");

/** Gate at τ, written exactly as device_condition_rot4d so a zero readout reproduces it bit for bit. */
__device__ __forceinline__ float device_rs_opacity(float base, float mu_t, float inv_var_t, float tau, float thr)
{
    const float dt = tau - mu_t;
    const float wt = exp(-0.5f * dt * dt * inv_var_t);
    const bool passes = (thr <= 0.f) || (wt >= thr);
    return passes ? base * wt : 0.f;
}

/** One conditioned Gaussian: sigma points at tau_ref and what moves them to another row. */
struct RsGaussian {
    float pts[7][3];
    float L[6]; // covariance factor
    float v[3];
    float base;
    float mu_t;
    float inv_var_t;
};

/** One camera's rows and the projection settings shared by every row. */
struct RsCamera {
    const float* row_vms; // [H, 4, 4]
    const float* dist;
    float fx, fy, cx, cy;
    int width, height;
    float tau_ref, tau_first, tau_step;
    float eps2d, near_plane, far_plane, prefilter_threshold;
};

__device__ __forceinline__ RsGaussian
device_rs_gaussian(const float mean[3], const float L[6], const float v[3], float base, float mu_t, float inv_var_t)
{
    RsGaussian g;
    device_sigma_points_from_factor(mean, L, UT_C_SPREAD, g.pts);
#pragma unroll
    for (int i = 0; i < 6; ++i)
        g.L[i] = L[i];
#pragma unroll
    for (int i = 0; i < 3; ++i)
        g.v[i] = v[i];
    g.base = base;
    g.mu_t = mu_t;
    g.inv_var_t = inv_var_t;
    return g;
}

/** The global-shutter UT projection at row r: sigma points moved to τ_r, row r's viewmat. False where it culls. */
template <ProjectionFamily F>
__device__ __forceinline__ bool device_rs_row_project(
    const RsGaussian& g,
    const RsCamera& c,
    int r,
    float& m2x,
    float& m2y,
    float& depth,
    int32_t& rx,
    int32_t& ry
)
{
    const float tau_r = c.tau_first + (float)r * c.tau_step;
    const float opac = device_rs_opacity(g.base, g.mu_t, g.inv_var_t, tau_r, c.prefilter_threshold);
    if (opac < ALPHA_THRESHOLD)
        return false;
    const float dtau = tau_r - c.tau_ref;
    float pts[7][3];
#pragma unroll
    for (int k = 0; k < 7; ++k)
#pragma unroll
        for (int i = 0; i < 3; ++i)
            pts[k][i] = g.pts[k][i] + g.v[i] * dtau;
    const float wm[7] = {UT_WM0, UT_WMI, UT_WMI, UT_WMI, UT_WMI, UT_WMI, UT_WMI};
    const float wc[7] = {UT_WC0, UT_WCI, UT_WCI, UT_WCI, UT_WCI, UT_WCI, UT_WCI};
    float ca, cb, cc;
    return device_ut_project<F>(
        pts,
        g.L,
        wm,
        wc,
        c.row_vms + r * 16,
        c.fx,
        c.fy,
        c.cx,
        c.cy,
        c.eps2d,
        opac,
        c.near_plane,
        c.far_plane,
        c.width,
        c.height,
        c.dist,
        m2x,
        m2y,
        ca,
        cb,
        cc,
        depth,
        rx,
        ry
    );
}

/** Tiles [lo, hi) that gsplat's isect_tiles assigns a box (centre, radius) along one axis. */
__device__ __forceinline__ void device_isect_tile_range(float centre, int32_t radius, int n_tiles, int& lo, int& hi)
{
    const float t = centre / (float)TILE_SIZE, tr = (float)radius / (float)TILE_SIZE;
    lo = min(max(0, (int)floorf(t - tr)), n_tiles);
    hi = min(max(0, (int)ceilf(t + tr)), n_tiles);
}

/**
 * Whether the global-shutter render at row r draws the Gaussian in tile (tile_x, row r's tile row), and that row's
 * horizontal tile range. Box and rasterizer both decide through this, so they agree.
 */
template <ProjectionFamily F>
__device__ __forceinline__ bool
device_rs_row_tiles(const RsGaussian& g, const RsCamera& c, int r, float& depth, float& m2y, int& tx0, int& tx1)
{
    float m2x;
    int32_t rx, ry;
    if (!device_rs_row_project<F>(g, c, r, m2x, m2y, depth, rx, ry))
        return false;
    const int n_tx = (c.width + TILE_SIZE - 1) / TILE_SIZE, n_ty = (c.height + TILE_SIZE - 1) / TILE_SIZE;
    int ty0, ty1;
    device_isect_tile_range(m2y, ry, n_ty, ty0, ty1);
    device_isect_tile_range(m2x, rx, n_tx, tx0, tx1);
    const int ty = r / TILE_SIZE;
    return ty >= ty0 && ty < ty1 && tx0 < tx1;
}

/** Rows [r0, r1] whose gate can reach the draw threshold, padded in τ by fp32 rounding; false when there are none. */
__device__ __forceinline__ bool device_rs_active_rows(const RsGaussian& g, const RsCamera& c, int& r0, int& r1)
{
    const float w_min = fmaxf(c.prefilter_threshold, ALPHA_THRESHOLD / g.base);
    if (!(w_min < 1.f))
        return false;
    if (c.tau_step <= 0.f) {
        r0 = 0;
        r1 = c.height - 1;
        return true;
    }
    const float dt_max = sqrtf(-2.f * logf(w_min) / g.inv_var_t);
    // tau_r = tau_first + r · tau_step rounds by ~ulp(τ); a few ulp of slack keep every row the gate passes.
    const float tau_last = c.tau_first + (float)(c.height - 1) * c.tau_step;
    const float tau_mag = fmaxf(fabsf(c.tau_first), fabsf(tau_last));
    const float slack = 8.f * FLT_EPSILON * (fabsf(g.mu_t) + dt_max + tau_mag);
    const float lo = fmaxf(floorf((g.mu_t - dt_max - slack - c.tau_first) / c.tau_step) - 1.f, 0.f);
    const float hi = fminf(ceilf((g.mu_t + dt_max + slack - c.tau_first) / c.tau_step) + 1.f, (float)(c.height - 1));
    if (lo > hi)
        return false;
    r0 = (int)lo;
    r1 = (int)hi;
    return true;
}

template <ProjectionFamily F>
__device__ __forceinline__ RsCamera device_rs_camera(
    const CameraViewDevice& cam,
    int cam_idx,
    float tau_ref,
    float tau_first,
    float tau_step,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold
)
{
    const float* K = cam.Ks + cam_idx * 9;
    return RsCamera{
        cam.viewmats + (size_t)cam_idx * cam.height * 16,
        cam.distortion + cam_idx * CameraOps<F>::n_slots,
        K[0],
        K[4],
        K[2],
        K[5],
        cam.width,
        cam.height,
        tau_ref,
        tau_first,
        tau_step,
        eps2d,
        near_plane,
        far_plane,
        prefilter_threshold
    };
}

/** Camera centre at the frame time (f = 0.5): row centres are linear in f, so two middle rows average to it. */
__device__ __forceinline__ void device_rs_frame_centre(const float* row_vms, int height, float out[3])
{
    const int lo = (height - 1) / 2, hi = height / 2;
    out[0] = out[1] = out[2] = 0.f;
    for (const int r : {lo, hi}) {
        const float* vm = row_vms + r * 16;
        const float tx = vm[3], ty = vm[7], tz = vm[11];
        out[0] -= 0.5f * (vm[0] * tx + vm[4] * ty + vm[8] * tz);
        out[1] -= 0.5f * (vm[1] * tx + vm[5] * ty + vm[9] * tz);
        out[2] -= 0.5f * (vm[2] * tx + vm[6] * ty + vm[10] * tz);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection: conditioning at tau_ref + per-row UT tile box + SH colour
// ─────────────────────────────────────────────────────────────────────────────

template <ProjectionFamily F, AppearanceKind M>
__launch_bounds__(256, 2) __global__ void ut_4d_rolling_shutter_projection_kernel(
    const float* __restrict__ quats_l,
    const float* __restrict__ quats_r,
    const float* __restrict__ scales_4d_log,
    const float* __restrict__ means,
    const float* __restrict__ mu_t,
    const float* __restrict__ opacities_in, // [N] raw logit
    const float* __restrict__ sh0,
    const float* __restrict__ shN,
    const float* __restrict__ sh_temporal,
    int sh_degree,
    int shN_stride,
    int n_temporal_terms,
    int temporal_stride,
    CameraViewDevice cam, // viewmats = row viewmats [C, H, 4, 4]
    float tau_ref,
    float tau_first,
    float tau_step,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    float* __restrict__ means2d_out,     // [C, N, 2]
    float* __restrict__ depths_out,      // [C, N]
    int32_t* __restrict__ radii_out,     // [C, N, 2]
    float* __restrict__ colors_out,      // [C, N, 3]
    float* __restrict__ means3d_out,     // [N, 3] at tau_ref
    float* __restrict__ cov_factors_out, // [N, 6]
    float* __restrict__ base_out,        // [N] sigmoid(opacity), ungated
    float* __restrict__ motion_out,      // [N, 4] (v, mu_t)
    float* __restrict__ inv_var_out,     // [N] 1 / cov_t_marg_safe
    int N
)
{
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= cam.C * N)
        return;
    const int g = global_idx % N;
    const int cam_idx = global_idx / N;
    const int out_idx = cam_idx * N + g;

    float mean[3], L[6], v[3], wt_ref, opac_ref, inv_var_t;
    device_condition_rot4d<float>(
        quats_l + g * 4,
        quats_r + g * 4,
        scales_4d_log + g * 4,
        means[g * 3 + 0],
        means[g * 3 + 1],
        means[g * 3 + 2],
        mu_t[g],
        opacities_in[g],
        tau_ref,
        cov_eps,
        0.f,
        mean[0],
        mean[1],
        mean[2],
        L,
        wt_ref,
        opac_ref,
        v,
        &inv_var_t
    );
    const float base = sigmoid(opacities_in[g]);
    const float mt = mu_t[g];

    if (cam_idx == 0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            means3d_out[g * 3 + i] = mean[i];
            motion_out[g * 4 + i] = v[i];
        }
#pragma unroll
        for (int i = 0; i < 6; ++i)
            cov_factors_out[g * 6 + i] = L[i];
        base_out[g] = base;
        motion_out[g * 4 + 3] = mt;
        inv_var_out[g] = inv_var_t;
    }

    const RsCamera c = device_rs_camera<
        F>(cam, cam_idx, tau_ref, tau_first, tau_step, eps2d, near_plane, far_plane, prefilter_threshold);
    radii_out[out_idx * 2 + 0] = 0;
    radii_out[out_idx * 2 + 1] = 0;
    int r0, r1;
    const RsGaussian rg = device_rs_gaussian(mean, L, v, base, mt, inv_var_t);
    if (device_rs_active_rows(rg, c, r0, r1)) {
        // Box: the tiles every row's global render assigns on that row's tile row, in tile units.
        int tx0 = INT_MAX, tx1 = INT_MIN, ty0 = INT_MAX, ty1 = INT_MIN;
        float depth = 0.f, best_gap = INFINITY;
        for (int r = r0; r <= r1; ++r) {
            float m2y, d;
            int a, b;
            if (!device_rs_row_tiles<F>(rg, c, r, d, m2y, a, b))
                continue;
            tx0 = min(tx0, a);
            tx1 = max(tx1, b);
            ty0 = min(ty0, r / TILE_SIZE);
            ty1 = max(ty1, r / TILE_SIZE + 1);
            // Sort key from the row whose own centre line is nearest the Gaussian's centre.
            const float gap = fabsf(m2y - ((float)r + 0.5f));
            if (gap < best_gap) {
                best_gap = gap;
                depth = d;
            }
        }
        if (tx0 < tx1) {
            // Centre on tile midlines, radius half the span: isect_tiles reproduces [tx0, tx1) × [ty0, ty1) exactly.
            means2d_out[out_idx * 2 + 0] = 0.5f * (float)((tx0 + tx1) * TILE_SIZE);
            means2d_out[out_idx * 2 + 1] = 0.5f * (float)((ty0 + ty1) * TILE_SIZE);
            radii_out[out_idx * 2 + 0] = (tx1 - tx0) * TILE_SIZE / 2;
            radii_out[out_idx * 2 + 1] = (ty1 - ty0) * TILE_SIZE / 2;
            depths_out[out_idx] = depth;
        }
    }

    float centre[3];
    device_rs_frame_centre(c.row_vms, c.height, centre);
    float r, gc, b;
    AppearanceOps<M>::eval(
        sh_degree,
        n_temporal_terms,
        temporal_stride,
        mean[0] - centre[0],
        mean[1] - centre[1],
        mean[2] - centre[2],
        tau_ref - mt,
        ShCoeffs{sh0 + g * 3, shN + g * shN_stride},
        (sh_temporal != nullptr && n_temporal_terms > 0) ? sh_temporal + g * temporal_stride : nullptr,
        r,
        gc,
        b
    );
    colors_out[out_idx * 3 + 0] = r;
    colors_out[out_idx * 3 + 1] = gc;
    colors_out[out_idx * 3 + 2] = b;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rasterization: per-row admission, ray, mean and gate
// ─────────────────────────────────────────────────────────────────────────────

// GaussianBatch, then per slot: motion (v, mu_t), 1 / var_t, admission bit per tile row.
constexpr size_t RS_BATCH_BYTES = GAUSSIAN_BATCH_BYTES + sizeof(float4) + sizeof(float) + sizeof(uint32_t);

template <ProjectionFamily F>
__launch_bounds__(TILE_SIZE* TILE_SIZE, 2) __global__ void rasterize_from_world_rolling_shutter_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    const float* __restrict__ means_ref,    // [N, 3]
    const float* __restrict__ cov_factors,  // [N, 6] packed lower L
    const float* __restrict__ base_opac,    // [N]
    const float* __restrict__ motion,       // [N, 4]
    const float* __restrict__ inv_var_t,    // [N]
    const float* __restrict__ colors,       // [C, N, 3]
    CameraViewDevice cam,                   // viewmats = row viewmats [C, H, 4, 4]
    const int32_t* __restrict__ valid_mask, // [C, H, W_words] bitpacked, nullptr ⇒ every pixel observed
    const uint32_t mask_words,
    float tau_ref,
    float tau_first,
    float tau_step,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const uint32_t tile_width,
    const uint32_t tile_height,
    float* __restrict__ render_colors, // [C, H, W, 3]
    float* __restrict__ render_alphas  // [C, H, W]
)
{
    auto block = cg::this_thread_block();
    const uint32_t W = cam.width, H = cam.height, C = cam.C;
    const uint32_t cam_id = block.group_index().x;
    const uint32_t tile_y = block.group_index().y;
    const uint32_t tile_x = block.group_index().z;
    const uint32_t tile_id = tile_y * tile_width + tile_x;
    const uint32_t py = tile_y * TILE_SIZE + block.thread_index().y;
    const uint32_t px = tile_x * TILE_SIZE + block.thread_index().x;
    const bool inside = (py < H && px < W);

    tile_offsets += cam_id * tile_height * tile_width;
    render_colors += cam_id * H * W * 3;
    render_alphas += cam_id * H * W;

    const RsCamera c = device_rs_camera<
        F>(cam, (int)cam_id, tau_ref, tau_first, tau_step, eps2d, near_plane, far_plane, prefilter_threshold);
    const uint32_t row = min(py, H - 1);
    const uint32_t row_bit = 1u << block.thread_index().y;
    const float* vm = c.row_vms + row * 16;
    const float* K_p = cam.Ks + cam_id * 9;
    const float tau_row = tau_first + (float)row * tau_step;
    const float dtau = tau_row - tau_ref;

    float ox, oy, oz, dx, dy, dz;
    const int32_t* vmask = (valid_mask == nullptr) ? nullptr : valid_mask + cam_id * H * mask_words;
    // Same gate as the global-shutter rasterizer: unobserved pixels stay empty.
    const bool ray_valid = device_pixel_ray<F>(vm, K_p, c.dist, (float)px, (float)py, ox, oy, oz, dx, dy, dz) &&
                           (!inside || device_pixel_observed(vmask, mask_words, px, py));
    bool done = !inside || !ray_valid;

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (cam_id == C - 1 && tile_id == tile_width * tile_height - 1) ? (int32_t)n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ __align__(16) int smem[];
    const GaussianBatch batch = device_gaussian_batch(smem, block_size);
    float4* s_motion = (float4*)&batch.l45[block_size]; // 16-byte aligned: block_size is a multiple of 4
    float* s_inv = (float*)&s_motion[block_size];
    uint32_t* s_rows = (uint32_t*)&s_inv[block_size];

    float T = 1.0f;
    float pix_out[3] = {};

    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size)
            break;
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + block.thread_rank();
        if (idx < (uint32_t)range_end) {
            const int32_t flat = flatten_ids[idx];
            const int32_t g = (int32_t)((uint32_t)flat % N);
            const uint32_t slot = block.thread_rank();
            device_store_gaussian(batch, slot, flat, g, means_ref, cov_factors, base_opac);
            const float4 mv = make_float4(motion[g * 4 + 0], motion[g * 4 + 1], motion[g * 4 + 2], motion[g * 4 + 3]);
            s_motion[slot] = mv;
            s_inv[slot] = inv_var_t[g];
            // Rows of this tile on which the global render at that row draws the Gaussian in this tile.
            const float v[3] = {mv.x, mv.y, mv.z};
            const RsGaussian rg =
                device_rs_gaussian(means_ref + g * 3, cov_factors + g * 6, v, base_opac[g], mv.w, inv_var_t[g]);
            uint32_t rows = 0;
            for (uint32_t k = 0; k < TILE_SIZE && tile_y * TILE_SIZE + k < H; ++k) {
                float d, m2y;
                int a, b;
                if (device_rs_row_tiles<F>(rg, c, (int)(tile_y * TILE_SIZE + k), d, m2y, a, b) && (int)tile_x >= a &&
                    (int)tile_x < b)
                    rows |= 1u << k;
            }
            s_rows[slot] = rows;
        }
        block.sync();

        const uint32_t batch_size = min(block_size, (uint32_t)(range_end - (int32_t)batch_start));
        for (uint32_t t = 0; t < batch_size && !done; ++t) {
            if (!(s_rows[t] & row_bit))
                continue;
            float4 mo;
            float Lf[6];
            device_load_gaussian(batch, t, mo, Lf);
            const float4 mv = s_motion[t];
            const float opac = device_rs_opacity(mo.w, mv.w, s_inv[t], tau_row, prefilter_threshold);

            float a, d2, r[3];
            device_ray_gaussian_mahal(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                mo.x + mv.x * dtau,
                mo.y + mv.y * dtau,
                mo.z + mv.z * dtau,
                Lf,
                a,
                r,
                d2
            );
            if (a <= 0.f)
                continue;
            const float max_response = __expf(-0.5f * d2);
            const float alpha = fminf(MAX_RASTER_ALPHA, opac * max_response);
            if (alpha < ALPHA_THRESHOLD || max_response <= MAX_RASTER_KERNEL_DENSITY)
                continue;
            const float next_T = T * (1.f - alpha);
            if (next_T <= T_RASTER_THRESHOLD) {
                done = true;
                break;
            }
            const float vis = alpha * T;
            const float* c_ptr = colors + batch.ids[t] * 3;
#pragma unroll
            for (int k = 0; k < 3; ++k)
                pix_out[k] += c_ptr[k] * vis;
            T = next_T;
        }
    }

    if (inside) {
        const uint32_t pix_id = py * W + px;
        render_alphas[pix_id] = 1.f - T;
#pragma unroll
        for (int k = 0; k < 3; ++k)
            render_colors[pix_id * 3 + k] = pix_out[k];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host entries
// ─────────────────────────────────────────────────────────────────────────────

static CameraViewDevice
rs_camera(const Tensor& row_viewmats, const Tensor& Ks, const Tensor& radial_coeffs, int family, int width, int height)
{
    const int C = (int)row_viewmats.size(0);
    TORCH_CHECK(
        row_viewmats.dim() == 4 && row_viewmats.size(1) == height && row_viewmats.size(2) == 4 &&
            row_viewmats.size(3) == 4,
        "row_viewmats must be [C, H, 4, 4]"
    );
    const int expected_stride = (family == FISHEYE_FAMILY) ? 4 : 12;
    TORCH_CHECK(
        radial_coeffs.dim() == 2 && radial_coeffs.size(0) == C && radial_coeffs.size(1) == expected_stride,
        "radial_coeffs must be [C, ",
        expected_stride,
        "] for this projection family"
    );
    TORCH_CHECK(Ks.dim() == 3 && Ks.size(0) == C && Ks.size(1) == 3 && Ks.size(2) == 3, "Ks must be [C, 3, 3]");
    TORCH_CHECK(
        row_viewmats.is_contiguous() && Ks.is_contiguous() && radial_coeffs.is_contiguous(),
        "camera tensors must be contiguous"
    );
    return CameraViewDevice{
        row_viewmats.data_ptr<float>(),
        Ks.data_ptr<float>(),
        radial_coeffs.data_ptr<float>(),
        C,
        (int)radial_coeffs.size(1),
        width,
        height
    };
}

/** Returns (means2d [C,N,2], depths [C,N], radii [C,N,2], colors [C,N,3], means [N,3], cov_factors [N,6], base [N],
 *  motion [N,4], inv_var_t [N]). */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> ut_4d_rolling_shutter_projection_fwd(
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
    const Tensor& row_viewmats,
    const Tensor& Ks,
    const Tensor& radial_coeffs,
    float tau_ref,
    float tau_first,
    float tau_step,
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
    TORCH_CHECK(tau_step >= 0.f, "tau_step must be >= 0 (rows exposed in order)");
    const CameraViewDevice cam = rs_camera(row_viewmats, Ks, radial_coeffs, family, image_width, image_height);
    const int N = (int)means.size(0);
    const int C = cam.C;

    auto opts_f = means.options();
    Tensor means2d = at::zeros({C, N, 2}, opts_f);
    Tensor depths = at::zeros({C, N}, opts_f);
    Tensor radii = at::zeros({C, N, 2}, opts_f.dtype(c10::kInt));
    Tensor colors = at::zeros({C, N, 3}, opts_f);
    Tensor means3d = at::zeros({N, 3}, opts_f);
    Tensor cov_factors = at::zeros({N, 6}, opts_f);
    Tensor base = at::zeros({N}, opts_f);
    Tensor motion = at::zeros({N, 4}, opts_f);
    Tensor inv_var = at::zeros({N}, opts_f);
    if (C * N == 0)
        return {means2d, depths, radii, colors, means3d, cov_factors, base, motion, inv_var};

    const Tensor sh_temporal_c = sh_temporal.contiguous(); // kept alive across the launch
    const float* sh_temporal_ptr = (sh_temporal_c.numel() > 0) ? sh_temporal_c.data_ptr<float>() : nullptr;
    const int block_size = 256;
    const int grid_size = (C * N + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_RS_PROJ(FAMILY)                                                                                         \
    ut_4d_rolling_shutter_projection_kernel<FAMILY, M><<<grid_size, block_size, 0, stream>>>(                          \
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
        tau_ref,                                                                                                       \
        tau_first,                                                                                                     \
        tau_step,                                                                                                      \
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
        base.data_ptr<float>(),                                                                                        \
        motion.data_ptr<float>(),                                                                                      \
        inv_var.data_ptr<float>(),                                                                                     \
        N                                                                                                              \
    )

    switch (family) {
    case PINHOLE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_RS_PROJ(PINHOLE_FAMILY));
        break;
    case FISHEYE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_RS_PROJ(FISHEYE_FAMILY));
        break;
    default:
        TORCH_CHECK(false, "unsupported projection family ", family);
    }
#undef LAUNCH_RS_PROJ

    return {means2d, depths, radii, colors, means3d, cov_factors, base, motion, inv_var};
}

/** Returns (render_colors [C, H, W, 3], render_alphas [C, H, W]). */
std::tuple<Tensor, Tensor> rasterize_from_world_rolling_shutter_fwd(
    const Tensor& means,
    const Tensor& cov_factors,
    const Tensor& base_opacities,
    const Tensor& motion,
    const Tensor& inv_var_t,
    const Tensor& colors,
    const Tensor& row_viewmats,
    const Tensor& Ks,
    const Tensor& radial_coeffs,
    float tau_ref,
    float tau_first,
    float tau_step,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    const Tensor& tile_offsets,
    const Tensor& flatten_ids,
    int image_width,
    int image_height,
    int family,
    const std::optional<Tensor>& valid_mask
)
{
    const CameraViewDevice cam = rs_camera(row_viewmats, Ks, radial_coeffs, family, image_width, image_height);
    const auto [vmask_ptr, mask_words] = device_valid_mask_ptr(valid_mask, cam.C, image_width, image_height);
    const int C = cam.C;
    const int N = (int)means.size(0);
    TORCH_CHECK(means.is_cuda() && means.dim() == 2 && means.size(1) == 3, "means must be a CUDA [N, 3] tensor");
    TORCH_CHECK(cov_factors.sizes() == at::IntArrayRef({N, 6}), "cov_factors must be [N, 6]");
    TORCH_CHECK(motion.sizes() == at::IntArrayRef({N, 4}), "motion must be [N, 4]");
    TORCH_CHECK(base_opacities.numel() == N && inv_var_t.numel() == N, "base_opacities and inv_var_t must be [N]");
    TORCH_CHECK(colors.sizes() == at::IntArrayRef({C, N, 3}), "colors must be [C, N, 3]");
    const int tile_height = (int)tile_offsets.size(1);
    const int tile_width = (int)tile_offsets.size(2);

    auto opts = means.options();
    Tensor render_colors = at::zeros({C, image_height, image_width, 3}, opts);
    Tensor render_alphas = at::zeros({C, image_height, image_width}, opts);

    const dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);
    const dim3 grid_dim(C, tile_height, tile_width);
    const size_t smem_bytes = RS_BATCH_BYTES * TILE_SIZE * TILE_SIZE;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_RS_RASTER(FAMILY)                                                                                       \
    rasterize_from_world_rolling_shutter_kernel<FAMILY><<<grid_dim, block_dim, smem_bytes, stream>>>(                  \
        N,                                                                                                             \
        (uint32_t)flatten_ids.size(0),                                                                                 \
        means.contiguous().data_ptr<float>(),                                                                          \
        cov_factors.contiguous().data_ptr<float>(),                                                                    \
        base_opacities.contiguous().data_ptr<float>(),                                                                 \
        motion.contiguous().data_ptr<float>(),                                                                         \
        inv_var_t.contiguous().data_ptr<float>(),                                                                      \
        colors.contiguous().data_ptr<float>(),                                                                         \
        cam,                                                                                                           \
        vmask_ptr,                                                                                                     \
        mask_words,                                                                                                    \
        tau_ref,                                                                                                       \
        tau_first,                                                                                                     \
        tau_step,                                                                                                      \
        eps2d,                                                                                                         \
        near_plane,                                                                                                    \
        far_plane,                                                                                                     \
        prefilter_threshold,                                                                                           \
        tile_offsets.contiguous().data_ptr<int32_t>(),                                                                 \
        flatten_ids.contiguous().data_ptr<int32_t>(),                                                                  \
        tile_width,                                                                                                    \
        tile_height,                                                                                                   \
        render_colors.data_ptr<float>(),                                                                               \
        render_alphas.data_ptr<float>()                                                                                \
    )

    switch (family) {
    case PINHOLE_FAMILY:
        LAUNCH_RS_RASTER(PINHOLE_FAMILY);
        break;
    case FISHEYE_FAMILY:
        LAUNCH_RS_RASTER(FISHEYE_FAMILY);
        break;
    default:
        TORCH_CHECK(false, "unsupported projection family ", family);
    }
#undef LAUNCH_RS_RASTER

    return {render_colors, render_alphas};
}
