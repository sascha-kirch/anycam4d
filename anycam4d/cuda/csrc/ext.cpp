/**
 * ext.cpp — pybind11 bindings for anycam4d CUDA extension.
 *
 * Exposes:
 *   launch_ewa_4d_fused_conditioned_projection_fwd_kernel / launch_ewa_4d_fused_conditioned_projection_bwd_kernel
 */

#include "common.h"
#include <optional>
#include <torch/extension.h>
#include <tuple>

using at::Tensor;

// Declaration (defined in stso_adam.cu)

void stso_adam_update(
    Tensor& param,
    const Tensor& param_grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    const Tensor& visibility,
    double lr,
    double b1,
    double b2,
    double eps
);

// Declaration (defined in ewa_4d_fused_conditioned_projection_fwd.cu)
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
);

// Declaration (defined in ewa_4d_fused_packed_conditioned_projection_fwd.cu)
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
);

// Declaration (defined in ewa_4d_fused_packed_conditioned_projection_bwd.cu)
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
launch_ewa_4d_fused_packed_conditioned_projection_bwd_kernel(
    const Tensor& d_means2d,
    const Tensor& d_conics,
    const Tensor& d_colors,
    const Tensor& d_opac,
    const Tensor& camera_ids,
    const Tensor& gaussian_ids,
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
);

// Declaration (defined in ewa_4d_fused_conditioned_projection_bwd.cu)
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
launch_ewa_4d_fused_conditioned_projection_bwd_kernel(
    const Tensor& d_means2d,
    const Tensor& d_conics,
    const Tensor& d_colors,
    const Tensor& d_opac,
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
);

// ─── Fused UT (conditioning + UT projection + SH4D) ──────────────────────────

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
);

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
);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> ut_4d_fused_projection_bwd(
    const Tensor& d_means_3d_cond,
    const Tensor& d_cov_factors,
    const Tensor& d_opacities,
    const Tensor& d_colors,
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
);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> ut_4d_fused_packed_projection_bwd(
    const Tensor& d_colors,
    const Tensor& d_means_3d_cond,
    const Tensor& d_cov_factors,
    const Tensor& d_opacities,
    const Tensor& camera_ids,
    const Tensor& gaussian_ids,
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
);

// ─── Depth-of-field forward projection (no backward) ─────────────────────────

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
);

// ─── Rolling-shutter forward render (no backward) ─────────────────────────────

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
);

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
);

// ─── World-space rasterization ────────────────────────────────────────────────

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
);

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
);

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
);

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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "anycam4d CUDA kernels";

    // ── Projection family enum (compile-time device dispatch tag) ─────────────
    // The UT projection / world-rasterization bindings take a ``family`` int that
    // selects the kernel template instantiation host-side. Mirrors ``ProjectionFamily``
    // in ``anycam4d/camera/registry.py`` (the Python registry owns the family→int
    // mapping). Exposed for parity assertions; bindings accept the raw int.
    py::enum_<ProjectionFamily>(
        m,
        "ProjectionFamily",
        py::arithmetic(),
        "Projection family — compile-time device dispatch tag.\n\n"
        "  PINHOLE_FAMILY (0): perspective divide + full OpenCV Brown-Conrady distortion.\n"
        "  FISHEYE_FAMILY (1): equidistant (OpenCV) fisheye, KB theta-polynomial.\n"
    )
        .value("PINHOLE_FAMILY", ProjectionFamily::PINHOLE_FAMILY)
        .value("FISHEYE_FAMILY", ProjectionFamily::FISHEYE_FAMILY)
        .export_values();

    // ── Appearance kind enum (compile-time colour-leaf dispatch tag) ──────────
    // The fused projection bindings take an ``appearance_kind`` int that selects the
    // kernel template instantiation host-side. Mirrors ``APPEARANCE_CUDA_ID`` in
    // ``anycam4d/primitives/appearance.py``. Exposed for parity assertions; bindings
    // accept the raw int.
    py::enum_<AppearanceKind>(
        m,
        "AppearanceKind",
        py::arithmetic(),
        "Appearance kind — compile-time colour-leaf dispatch tag.\n\n"
        "  Additive (0): spatial SH plus a direction-independent temporal Fourier term.\n"
        "  Multiplicative (1): published 4D SH — every spatial harmonic times its temporal cosine.\n"
    )
        .value("Additive", AppearanceKind::Additive)
        .value("Multiplicative", AppearanceKind::Multiplicative)
        .export_values();

    // The multiplicative basis allocates this many temporal orders; Python sizes sh_spacetime by its
    // own copy and the kernel divides the stride by this one, so the two must agree.
    m.attr("MAX_TEMPORAL_DEGREE") = MAX_TEMPORAL_DEGREE;

    m.def(
        "launch_ewa_4d_fused_conditioned_projection_fwd_kernel",
        &launch_ewa_4d_fused_conditioned_projection_fwd_kernel,
        "Fused forward: 4D conditioning + 4D SH + world-to-cam + EWA projection.\n\n"
        "One thread per (Gaussian, Camera) pair.  No intermediate tensors between stages.\n\n"
        "Args:\n"
        "    quats_l, quats_r  : [N, 4] left/right quaternions (wxyz, unnormalized).\n"
        "    scales_4d_log         : [N, 4] log-space 4D scales.\n"
        "    means             : [N, 3] 3D spatial means.\n"
        "    mu_t              : [N]   temporal means.\n"
        "    opacities         : [N]   raw logit opacities.\n"
        "    sh0, shN          : [N, 1, 3] and [N, K_max - 1, 3] SH coefficients, read in place.\n"
        "    sh_temporal       : [N, T, 3] temporal coefficients of the appearance model, or empty.\n"
        "    sh_degree         : int, spatial SH degree (0..3).\n"
        "    appearance_kind   : int, AppearanceKind selecting the colour leaf.\n"
        "    n_temporal_terms  : int, active terms of that leaf (0 = spatial SH only).\n"
        "    temporal_stride   : int, floats per Gaussian in sh_temporal.\n"
        "    viewmats          : [C, 4, 4] world-to-camera matrices (row-major).\n"
        "    Ks                : [C, 3, 3] camera intrinsics (row-major).\n"
        "    dist              : [C, n_slots] lens distortion, or empty for an undistorted pinhole.\n"
        "    family            : int, ProjectionFamily of dist (ignored when empty).\n"
        "    tau               : float, query timestamp.\n"
        "    cov_eps           : float, PSD floor for the cov_3d_cond diagonal.\n"
        "    eps2d             : float, blur added to 2D covariance diagonal.\n"
        "    near_plane        : float, near clipping depth.\n"
        "    far_plane         : float, far clipping depth.\n"
        "    prefilter_threshold: float, cull Gaussians with w_t < threshold.\n"
        "    image_width       : int, image width in pixels.\n"
        "    image_height      : int, image height in pixels.\n\n"
        "Returns: (means2d [C,N,2], conics [C,N,3], colors [C,N,3], opacities [C,N],\n"
        "          depths [C,N], radii [C,N,2 int32], means_3d_cond [N,3], wt [N])",
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("dist"),
        py::arg("family"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("calc_compensations")
    );

    m.def(
        "launch_ewa_4d_fused_conditioned_projection_bwd_kernel",
        &launch_ewa_4d_fused_conditioned_projection_bwd_kernel,
        "Fused backward: gradients through EWA projection → camera transform → SH → conditioning.\n\n"
        "All forward intermediates are recomputed in registers from saved inputs.\n"
        "Gradients are accumulated via atomicAdd across cameras.\n\n"
        "Args:\n"
        "    d_means2d        : [C, N, 2] upstream gradient w.r.t. projected 2D means.\n"
        "    d_conics         : [C, N, 3] upstream gradient w.r.t. conics.\n"
        "    d_colors         : [C, N, 3] upstream gradient w.r.t. colors.\n"
        "    d_opac           : [C, N]   upstream gradient w.r.t. effective opacities.\n"
        "    quats_l, quats_r : [N, 4] saved forward inputs (unnormalized).\n"
        "    scales_4d_log        : [N, 4] saved forward inputs (log-space).\n"
        "    means            : [N, 3] saved forward inputs.\n"
        "    mu_t             : [N]   saved forward inputs.\n"
        "    opacities        : [N]   saved forward inputs (logit).\n"
        "    sh0, shN         : saved forward inputs.\n"
        "    sh_temporal      : [N, T, 3] saved forward inputs, or empty tensor.\n"
        "    sh_degree        : int, spatial SH degree.\n"
        "    appearance_kind  : int, AppearanceKind selecting the colour leaf.\n"
        "    n_temporal_terms : int, active terms of that leaf.\n"
        "    temporal_stride  : int, floats per Gaussian in sh_temporal.\n"
        "    viewmats         : [C, 4, 4] saved forward inputs.\n"
        "    Ks               : [C, 3, 3] saved forward inputs.\n"
        "    tau, cov_eps, eps2d, near_plane, far_plane, prefilter_threshold: forward scalars.\n"
        "    image_width, image_height: image dimensions.\n"
        "    calc_compensations: bool, if True apply Mip-Splatting compensation to opacities.\n\n"
        "Returns: (d_quats_l [N,4], d_quats_r [N,4], d_scales_4d_log [N,4], d_means [N,3],\n"
        "          d_mu_t [N], d_opacities [N], d_sh0 [N,1,3], d_shN [N,K_max-1,3], d_sh_temporal [N,T,3])",
        py::arg("d_means2d"),
        py::arg("d_conics"),
        py::arg("d_colors"),
        py::arg("d_opac"),
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("dist"),
        py::arg("family"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("calc_compensations")
    );

    m.def(
        "launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel",
        &launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel,
        "Packed fused forward: two-pass count+scatter for 4D conditioning + SH + EWA projection.\n\n"
        "Returns packed outputs containing only visible (camera, Gaussian) pairs.\n\n"
        "Returns: (camera_ids [nnz], gaussian_ids [nnz], indptr [C+1],\n"
        "          means2d [nnz,2], conics [nnz,3], colors [nnz,3], opacities [nnz],\n"
        "          depths [nnz], radii [nnz,2 int32], means_3d_cond [N,3], wt [N])",
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("dist"),
        py::arg("family"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("calc_compensations")
    );

    m.def(
        "ut_4d_fused_packed_projection_fwd",
        &ut_4d_fused_packed_projection_fwd,
        "Packed fused forward: two-pass count+scatter for 4D conditioning + UT projection + SH.\n\n"
        "Tile-assignment outputs (means2d, conics, depths, radii) are packed to [nnz, ...].\n"
        "Rasterization inputs (colors, means_3d_cond, cov_factors, opacities) remain dense.\n\n"
        "Returns: (camera_ids [nnz], gaussian_ids [nnz], indptr [C+1],\n"
        "          means2d [nnz,2], conics [nnz,3], depths [nnz], radii [nnz,2 int32],\n"
        "          colors [C,N,3], means_3d_cond [N,3], cov_factors [N,6], opacities [N], wt [N])",
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("radial_coeffs"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("family")
    );

    m.def(
        "ut_4d_fused_projection_fwd",
        &ut_4d_fused_projection_fwd,
        "Fused forward: 4D conditioning + UT projection + 4D SH color eval.\n\n"
        "Conditioning, UT projection and per-camera SH in one pass.  Returns both no-grad tile-assignment outputs\n"
        "(means2d, conics, depths, radii) and differentiable outputs (means_3d_cond, cov_factors, opacities,\n"
        "colors, wt) consumed by RasterizeWorldFunction.\n\n"
        "Returns: (means2d [C,N,2], conics [C,N,3], depths [C,N], radii [C,N,2 int32],\n"
        "          colors [C,N,3], means_3d_cond [N,3], cov_factors [N,6], opacities [N], wt [N])",
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("radial_coeffs"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("family")
    );

    m.def(
        "ut_4d_fused_projection_bwd",
        &ut_4d_fused_projection_bwd,
        "Fused backward for ut_4d_fused_projection_fwd.\n\n"
        "Backpropagates only through the differentiable outputs (means_3d_cond, cov_factors, opacities,\n"
        "colors); the UT 2D projection outputs are stop-gradient sinks.\n\n"
        "Returns: (d_quats_l [N,4], d_quats_r [N,4], d_scales_4d_log [N,4], d_means [N,3],\n"
        "          d_mu_t [N], d_opacities [N], d_sh0 [N,1,3], d_shN [N,K_max-1,3], d_sh_temporal [N,T,3])",
        py::arg("d_means_3d_cond"),
        py::arg("d_cov_factors"),
        py::arg("d_opacities"),
        py::arg("d_colors"),
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("prefilter_threshold")
    );

    m.def(
        "ut_4d_fused_packed_projection_bwd",
        &ut_4d_fused_packed_projection_bwd,
        "Packed fused backward for ut_4d_fused_packed_projection_fwd.\n\n"
        "Parallelizes over nnz visible (camera, Gaussian) pairs. Per-Gaussian upstreams\n"
        "(d_means_3d_cond, d_cov_factors, d_opacities) are divided by vis_count[n] per thread to\n"
        "avoid C_vis-fold over-counting. SH gradients accumulate via atomicAdd.\n\n"
        "Returns: (d_quats_l [N,4], d_quats_r [N,4], d_scales_4d_log [N,4], d_means [N,3],\n"
        "          d_mu_t [N], d_opacities [N], d_sh0 [N,1,3], d_shN [N,K_max-1,3], d_sh_temporal [N,T,3])",
        py::arg("d_colors"),
        py::arg("d_means_3d_cond"),
        py::arg("d_cov_factors"),
        py::arg("d_opacities"),
        py::arg("camera_ids"),
        py::arg("gaussian_ids"),
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("prefilter_threshold")
    );

    m.def(
        "ut_4d_depth_of_field_projection_fwd",
        &ut_4d_depth_of_field_projection_fwd,
        "Thin-lens UT projection for one camera: scene sheared by aperture offset (offset_x, offset_y) and each\n"
        "Gaussian dilated by a disk of blur_radius at its depth, focus plane z = focus. Forward only.\n\n"
        "Returns: (means2d [1,N,2], depths [1,N], radii [1,N,2 int32], colors [1,N,3], means [N,3], cov_factors "
        "[N,6],\n"
        "          opacities [N])",
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("radial_coeffs"),
        py::arg("focus"),
        py::arg("blur_radius"),
        py::arg("offset_x"),
        py::arg("offset_y"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("family")
    );

    m.def(
        "ut_4d_rolling_shutter_projection_fwd",
        &ut_4d_rolling_shutter_projection_fwd,
        "Rolling-shutter UT projection: conditioning at tau_ref, tile box swept over the rows (row r at\n"
        "tau_first + r * tau_step, viewmat row_viewmats[c, r]), SH colour at tau_ref. Forward only.\n\n"
        "Returns: (means2d [C,N,2], depths [C,N], radii [C,N,2 int32], colors [C,N,3], means [N,3],\n"
        "          cov_factors [N,6], base_opacities [N], motion [N,4] (v, mu_t), inv_var_t [N])",
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("row_viewmats"),
        py::arg("Ks"),
        py::arg("radial_coeffs"),
        py::arg("tau_ref"),
        py::arg("tau_first"),
        py::arg("tau_step"),
        py::arg("cov_eps"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("family")
    );

    m.def(
        "rasterize_from_world_rolling_shutter_fwd",
        &rasterize_from_world_rolling_shutter_fwd,
        "Rolling-shutter world-space rasterization: per-row admission (the global UT projection at that row), ray, "
        "mean\n"
        "(means + v * dtau) and gate. Forward only.\n\n"
        "Returns: (render_colors [C,H,W,3], render_alphas [C,H,W])",
        py::arg("means"),
        py::arg("cov_factors"),
        py::arg("base_opacities"),
        py::arg("motion"),
        py::arg("inv_var_t"),
        py::arg("colors"),
        py::arg("row_viewmats"),
        py::arg("Ks"),
        py::arg("radial_coeffs"),
        py::arg("tau_ref"),
        py::arg("tau_first"),
        py::arg("tau_step"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("tile_offsets"),
        py::arg("flatten_ids"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("family"),
        py::arg("valid_mask") = std::optional<Tensor>()
    );

    m.def(
        "ut_4d_dispersion_projection_fwd",
        &ut_4d_dispersion_projection_fwd,
        "Forward-only UT projection for chromatic aberration: one pose, L lenses (Ks [L, 3, 3], radial_coeffs [L, P]);\n"
        "conditions and colours once, tile box = union of the lenses' UT boxes. Returns (means2d [1,N,2], depths [1,N],\n"
        "radii [1,N,2], colors [1,N,3], means [N,3], cov_factors [N,6], opacities [N])."
    );

    m.def(
        "rasterize_from_world_dispersion_fwd",
        &rasterize_from_world_dispersion_fwd,
        "Forward-only lateral chromatic aberration: 3S rays per pixel through per-wavelength lenses (Ks [3S, 3, 3],\n"
        "radial_coeffs [3S, P]) from one pose, ray r composited into channel r / S. Returns (colors [1, H, W, 3],\n"
        "alphas [1, H, W, 3]).",
        py::arg("means_3d_cond"),
        py::arg("cov_factors"),
        py::arg("opacities"),
        py::arg("colors"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("radial_coeffs"),
        py::arg("tile_offsets"),
        py::arg("flatten_ids"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("family"),
        py::arg("valid_mask") = py::none()
    );

    m.def(
        "rasterize_from_world_fwd",
        &rasterize_from_world_fwd,
        "Forward world-space rasterization using the precision-matrix Mahalanobis distance.\n\n"
        "Args:\n"
        "    means_3d_cond       : [N, 3] conditioned 3D Gaussian means.\n"
        "    cov_factors        : [N, 6] packed lower covariance factors L, Σ = L Lᵀ.\n"
        "    opacities     : [N]   effective opacities.\n"
        "    colors        : [C, N, CDIM] per-camera colors.\n"
        "    viewmats      : [C, 4, 4] world-to-camera matrices.\n"
        "    Ks            : [C, 3, 3] camera intrinsics.\n"
        "    radial_coeffs : [C, P] fixed-width OpenCV distortion (P = 4 fisheye / 12 pinhole-family; fisheye "
        "reads first 4).\n"
        "    tile_offsets  : [C, tile_height, tile_width].\n"
        "    flatten_ids   : [n_isects].\n"
        "    image_width   : int, image width in pixels.\n"
        "    image_height  : int, image height in pixels.\n"
        "    family        : int, projection family PINHOLE_FAMILY=0 / FISHEYE_FAMILY=1.\n"
        "    valid_mask    : optional [C, H, (W+31)//32] int32 bitpacked observation mask (set bit =\n"
        "                    observed). None means every pixel was observed.\n\n"
        "Returns: (render_colors [C,H,W,CDIM], render_alphas [C,H,W], last_ids [C,H,W])",
        py::arg("means_3d_cond"),
        py::arg("cov_factors"),
        py::arg("opacities"),
        py::arg("colors"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("radial_coeffs"),
        py::arg("tile_offsets"),
        py::arg("flatten_ids"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("family"),
        py::arg("valid_mask") = std::optional<Tensor>()
    );

    m.def(
        "rasterize_from_world_bwd",
        &rasterize_from_world_bwd,
        "Backward world-space rasterization: gradients w.r.t. means_3d_cond, cov_factors, opacities, colors.\n\n"
        "Args:\n"
        "    means_3d_cond, cov_factors, opacities, colors: same as forward.\n"
        "    viewmats, Ks             : same as forward.\n"
        "    radial_coeffs            : [C, P] fixed-width OpenCV distortion (P = 4 fisheye / 12 "
        "pinhole-family; fisheye reads first 4).\n"
        "    tile_offsets, flatten_ids: same as forward.\n"
        "    render_alphas            : [C, H, W] from forward.\n"
        "    last_ids                 : [C, H, W] from forward.\n"
        "    dL_drender_colors        : [C, H, W, CDIM] upstream gradient.\n"
        "    dL_drender_alphas        : [C, H, W] upstream gradient.\n"
        "    family                   : int, projection family PINHOLE_FAMILY=0 / FISHEYE_FAMILY=1.\n"
        "    valid_mask               : optional bitpacked observation mask, same as forward.\n\n"
        "Returns: (dL_dmeans3d [N,3], dL_dcov_factors [N,6], dL_dopacities [N], dL_dcolors [C,N,CDIM])",
        py::arg("means_3d_cond"),
        py::arg("cov_factors"),
        py::arg("opacities"),
        py::arg("colors"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("radial_coeffs"),
        py::arg("tile_offsets"),
        py::arg("flatten_ids"),
        py::arg("render_alphas"),
        py::arg("last_ids"),
        py::arg("dL_drender_colors"),
        py::arg("dL_drender_alphas"),
        py::arg("family"),
        py::arg("valid_mask") = std::optional<Tensor>()
    );

    m.def(
        "launch_ewa_4d_fused_packed_conditioned_projection_bwd_kernel",
        &launch_ewa_4d_fused_packed_conditioned_projection_bwd_kernel,
        "Packed fused backward: gradients through EWA + camera + SH + conditioning.\n\n"
        "Parallelizes over nnz (visible pairs only). Uses camera_ids/gaussian_ids\n"
        "to look up original parameters. Accumulates gradients via atomicAdd.\n\n"
        "Returns: (d_quats_l [N,4], d_quats_r [N,4], d_scales_4d_log [N,4], d_means [N,3],\n"
        "          d_mu_t [N], d_opacities [N], d_sh0 [N,1,3], d_shN [N,K_max-1,3], d_sh_temporal [N,T,3])",
        py::arg("d_means2d"),
        py::arg("d_conics"),
        py::arg("d_colors"),
        py::arg("d_opac"),
        py::arg("camera_ids"),
        py::arg("gaussian_ids"),
        py::arg("quats_l"),
        py::arg("quats_r"),
        py::arg("scales_4d_log"),
        py::arg("means"),
        py::arg("mu_t"),
        py::arg("opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("sh_temporal"),
        py::arg("sh_degree"),
        py::arg("appearance_kind"),
        py::arg("n_temporal_terms"),
        py::arg("temporal_stride"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("dist"),
        py::arg("family"),
        py::arg("tau"),
        py::arg("cov_eps"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("prefilter_threshold"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("calc_compensations")
    );

    m.def(
        "stso_adam_update",
        &stso_adam_update,
        "Fused selective Adam update (in-place, fp32-only), without bias correction.\n\n"
        "Updates only the active primitives (visibility[g_idx]); inactive rows keep param and both\n"
        "moments bit-exact.\n\n"
        "Args:\n"
        "    param          : [N, ...] parameter, mutated in place.\n"
        "    grad           : [N, ...] gradient.\n"
        "    exp_avg        : [N, ...] first-moment EMA, mutated in place.\n"
        "    exp_avg_sq     : [N, ...] second-moment EMA, mutated in place.\n"
        "    visibility     : [N] bool per-primitive active mask.\n"
        "    lr, b1, b2, eps: Adam hyperparameters (eps outside the sqrt).",
        py::arg("param"),
        py::arg("grad"),
        py::arg("exp_avg"),
        py::arg("exp_avg_sq"),
        py::arg("visibility"),
        py::arg("lr"),
        py::arg("b1"),
        py::arg("b2"),
        py::arg("eps")
    );
}
