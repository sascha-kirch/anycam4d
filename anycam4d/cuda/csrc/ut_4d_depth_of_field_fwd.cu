/**
 * Thin-lens depth of field for 4D Gaussians: forward-only UT projection, drawn by the shared world rasterizer.
 *
 * Aperture sample o = the scene sheared in camera coordinates, p' = p - o (1 - z/d) laterally, through the unchanged
 * camera. A blurred pass adds the aperture disk's covariance at the conditional mean depth and keeps the ray
 * footprint's integral.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

#include "conditioning.cuh"
#include "sh4d.cuh"
#include "ut_4d.cuh"

using at::Tensor;

/** vᵀ adj(Σ) v: for unit v, the determinant of Σ compressed onto the plane ⊥ v (the ray footprint). */
__device__ __forceinline__ float device_footprint_det(const float S[6], const float v[3])
{
    const float a00 = S[3] * S[5] - S[4] * S[4];
    const float a01 = S[2] * S[4] - S[1] * S[5];
    const float a02 = S[1] * S[4] - S[2] * S[3];
    const float a11 = S[0] * S[5] - S[2] * S[2];
    const float a12 = S[1] * S[2] - S[0] * S[4];
    const float a22 = S[0] * S[3] - S[1] * S[1];
    return v[0] * v[0] * a00 + v[1] * v[1] * a11 + v[2] * v[2] * a22 +
           2.f * (v[0] * v[1] * a01 + v[0] * v[2] * a02 + v[1] * v[2] * a12);
}

template <ProjectionFamily F, AppearanceKind M>
__launch_bounds__(256, 2) __global__ void ut_4d_depth_of_field_projection_kernel(
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
    CameraViewDevice cam, // C == 1
    float focus,
    float blur_radius,
    float offset_x,
    float offset_y,
    float tau,
    float cov_eps,
    float eps2d,
    float near_plane,
    float far_plane,
    float prefilter_threshold,
    float* __restrict__ means2d_out,     // [N, 2]
    float* __restrict__ depths_out,      // [N]
    int32_t* __restrict__ radii_out,     // [N, 2]
    float* __restrict__ colors_out,      // [N, 3]
    float* __restrict__ means3d_out,     // [N, 3]
    float* __restrict__ cov_factors_out, // [N, 6] packed lower L
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

    const float* vm = cam.viewmats;
    const float R[3][3] = {{vm[0], vm[1], vm[2]}, {vm[4], vm[5], vm[6]}, {vm[8], vm[9], vm[10]}};
    const float Rt[3][3] = {{vm[0], vm[4], vm[8]}, {vm[1], vm[5], vm[9]}, {vm[2], vm[6], vm[10]}};
    const float t[3] = {vm[3], vm[7], vm[11]};
    const float p[3] = {
        R[0][0] * mx + R[0][1] * my + R[0][2] * mz + t[0],
        R[1][0] * mx + R[1][1] * my + R[1][2] * mz + t[1],
        R[2][0] * mx + R[2][1] * my + R[2][2] * mz + t[2],
    };
    const float z = p[2];

    // Camera-frame factor B = S R L: shear by the aperture offset; the aperture disk at this depth is added below.
    const float shear[3][3] = {{1.f, 0.f, offset_x / focus}, {0.f, 1.f, offset_y / focus}, {0.f, 0.f, 1.f}};
    float RL[3][3], Lm[3][3], B[3][3];
    device_unpack_lower(L, Lm);
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            RL[i][j] = R[i][0] * Lm[0][j] + R[i][1] * Lm[1][j] + R[i][2] * Lm[2][j];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            B[i][j] = shear[i][0] * RL[0][j] + shear[i][1] * RL[1][j] + shear[i][2] * RL[2][j];
    float cov_dof[6];
    cov_dof[0] = B[0][0] * B[0][0] + B[0][1] * B[0][1] + B[0][2] * B[0][2];
    cov_dof[1] = B[0][0] * B[1][0] + B[0][1] * B[1][1] + B[0][2] * B[1][2];
    cov_dof[2] = B[0][0] * B[2][0] + B[0][1] * B[2][1] + B[0][2] * B[2][2];
    cov_dof[3] = B[1][0] * B[1][0] + B[1][1] * B[1][1] + B[1][2] * B[1][2];
    cov_dof[4] = B[1][0] * B[2][0] + B[1][1] * B[2][1] + B[1][2] * B[2][2];
    cov_dof[5] = B[2][0] * B[2][0] + B[2][1] * B[2][1] + B[2][2] * B[2][2];
    const float lift = 1.f - z / focus;
    const float p_dof[3] = {p[0] - offset_x * lift, p[1] - offset_y * lift, z};

    float scale = 1.f, disk_sd = 0.f;
    if (blur_radius > 0.f) {
        const float r = blur_radius * lift;
        // Uniform disk of radius r: variance r²/4 per axis. The rasterizer cuts each Gaussian at response d, which
        // keeps kappa = 1 - k d / (1 - d) of a 2D Gaussian's variance (k = -ln d); dividing by it gives the drawn blur
        // the disk's second moment.
        const float d = MAX_RASTER_KERNEL_DENSITY;
        const float kappa = 1.f - (-__logf(d)) * d / (1.f - d);
        const float var = 0.25f * r * r / kappa;
        const float inv_len = rsqrtf(fmaxf(p_dof[0] * p_dof[0] + p_dof[1] * p_dof[1] + z * z, 1e-20f));
        const float v[3] = {p_dof[0] * inv_len, p_dof[1] * inv_len, z * inv_len};
        const float det_sharp = device_footprint_det(cov_dof, v);
        cov_dof[0] += var;
        cov_dof[3] += var;
        disk_sd = sqrtf(var);
        scale = sqrtf(det_sharp / fmaxf(device_footprint_det(cov_dof, v), 1e-30f));
    }
    opacity *= scale;

    // Back to world: Σ_world = F Fᵀ, F = Rᵀ [B | √var e_x, √var e_y]; its factor by QR of Fᵀ.
    float Ft[5][3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int c = 0; c < 3; ++c)
            Ft[c][i] = Rt[i][0] * B[0][c] + Rt[i][1] * B[1][c] + Rt[i][2] * B[2][c];
        Ft[3][i] = Rt[i][0] * disk_sd;
        Ft[4][i] = Rt[i][1] * disk_sd;
    }
    float L_world[6];
    device_lower_factor_qr<float, 5>(Ft, L_world);
    const float q[3] = {p_dof[0] - t[0], p_dof[1] - t[1], p_dof[2] - t[2]};
    const float m_world[3] = {
        Rt[0][0] * q[0] + Rt[0][1] * q[1] + Rt[0][2] * q[2],
        Rt[1][0] * q[0] + Rt[1][1] * q[1] + Rt[1][2] * q[2],
        Rt[2][0] * q[0] + Rt[2][1] * q[1] + Rt[2][2] * q[2],
    };

#pragma unroll
    for (int i = 0; i < 3; ++i)
        means3d_out[g * 3 + i] = m_world[i];
#pragma unroll
    for (int i = 0; i < 6; ++i)
        cov_factors_out[g * 6 + i] = L_world[i];
    opacities_out[g] = opacity;

    radii_out[g * 2 + 0] = 0;
    radii_out[g * 2 + 1] = 0;
    if (opacity >= ALPHA_THRESHOLD) {
        float wm[7] = {UT_WM0, UT_WMI, UT_WMI, UT_WMI, UT_WMI, UT_WMI, UT_WMI};
        float wc[7] = {UT_WC0, UT_WCI, UT_WCI, UT_WCI, UT_WCI, UT_WCI, UT_WCI};
        float pts[7][3];
        device_sigma_points_from_factor(m_world, L_world, UT_C_SPREAD, pts);

        const float* K = cam.Ks;
        float mean2d_x, mean2d_y, conic_a, conic_b, conic_c, depth_unused;
        int32_t radius_x, radius_y;
        const bool visible = device_ut_project<F>(
            pts,
            L_world,
            wm,
            wc,
            vm,
            K[0],
            K[4],
            K[2],
            K[5],
            eps2d,
            opacity,
            near_plane,
            far_plane,
            cam.width,
            cam.height,
            cam.distortion,
            mean2d_x,
            mean2d_y,
            conic_a,
            conic_b,
            conic_c,
            depth_unused,
            radius_x,
            radius_y
        );
        if (visible) {
            means2d_out[g * 2 + 0] = mean2d_x;
            means2d_out[g * 2 + 1] = mean2d_y;
            // Sort key: distance from the aperture point to the unsheared mean, as the moved camera sorts.
            const float dx = p[0] - offset_x, dy = p[1] - offset_y;
            depths_out[g] = sqrtf(dx * dx + dy * dy + z * z);
            radii_out[g * 2 + 0] = radius_x;
            radii_out[g * 2 + 1] = radius_y;
        }
    }

    // Colour seen from the aperture point o (camera frame) toward the unsheared mean: the moved camera's view.
    const float a[3] = {
        Rt[0][0] * (offset_x - t[0]) + Rt[0][1] * (offset_y - t[1]) + Rt[0][2] * (-t[2]),
        Rt[1][0] * (offset_x - t[0]) + Rt[1][1] * (offset_y - t[1]) + Rt[1][2] * (-t[2]),
        Rt[2][0] * (offset_x - t[0]) + Rt[2][1] * (offset_y - t[1]) + Rt[2][2] * (-t[2]),
    };
    float r, gr, b;
    AppearanceOps<M>::eval(
        sh_degree,
        n_temporal_terms,
        temporal_stride,
        mx - a[0],
        my - a[1],
        mz - a[2],
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

/** Returns (means2d [1,N,2], depths [1,N], radii [1,N,2], colors [1,N,3], means [N,3], cov_factors [N,6], opacities
 * [N]). */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> ut_4d_depth_of_field_projection_fwd(
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
    float focus,
    float blur_radius,
    float offset_x,
    float offset_y,
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
    TORCH_CHECK(focus > 0.f, "focus must be > 0");
    TORCH_CHECK(blur_radius >= 0.f, "blur_radius must be >= 0");
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(0) == 1, "depth of field renders one camera: viewmats [1, 4, 4]");
    TORCH_CHECK(Ks.dim() == 3 && Ks.size(0) == 1, "Ks must be [1, 3, 3]");
    const int expected_stride = (family == FISHEYE_FAMILY) ? 4 : 12;
    TORCH_CHECK(
        radial_coeffs.dim() == 2 && radial_coeffs.size(0) == 1 && radial_coeffs.size(1) == expected_stride,
        "radial_coeffs must be [1, ",
        expected_stride,
        "] for this projection family"
    );
    TORCH_CHECK(
        viewmats.is_contiguous() && Ks.is_contiguous() && radial_coeffs.is_contiguous(),
        "camera tensors must be contiguous"
    );
    const CameraViewDevice cam{
        viewmats.data_ptr<float>(),
        Ks.data_ptr<float>(),
        radial_coeffs.data_ptr<float>(),
        1,
        expected_stride,
        image_width,
        image_height
    };
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

    const Tensor sh_temporal_c = sh_temporal.contiguous(); // kept alive across the launch
    const float* sh_temporal_ptr = (sh_temporal_c.numel() > 0) ? sh_temporal_c.data_ptr<float>() : nullptr;
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_DOF_PROJ(FAMILY)                                                                                        \
    ut_4d_depth_of_field_projection_kernel<FAMILY, M><<<grid_size, block_size, 0, stream>>>(                           \
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
        focus,                                                                                                         \
        blur_radius,                                                                                                   \
        offset_x,                                                                                                      \
        offset_y,                                                                                                      \
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
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_DOF_PROJ(PINHOLE_FAMILY));
        break;
    case FISHEYE_FAMILY:
        ANYCAM4D_DISPATCH_APPEARANCE(appearance_kind, LAUNCH_DOF_PROJ(FISHEYE_FAMILY));
        break;
    default:
        TORCH_CHECK(false, "unsupported projection family ", family);
    }
#undef LAUNCH_DOF_PROJ

    return {means2d, depths, radii, colors, means3d, cov_factors, opac};
}
