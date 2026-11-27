/**
 * Unscented projection of conditioned 4D Gaussians.
 *
 * Sigma points μ ± sqrt(3 + λ) · L[:, i] with Σ = L Lᵀ; identical to gsplat's R diag(s) points under
 * linear projection. Pipeline: 7 sigma points from L, project through CameraOps<F>, reconstruct
 * 2D mean and covariance with UT weights, then conic, sort key and radii.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "camera_ops.cuh"
#include "common.h"
#include "linalg.cuh"
#include "ut_constants.cuh" // paper-frozen UT α/β/κ + derived spread/weights (constexpr)

// ─────────────────────────────────────────────────────────────────────────────
// Sigma points from the covariance factor
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 7 sigma points σ₀ = μ, σᵢ = μ + c·L[:, i-1], σᵢ₊₃ = μ − c·L[:, i-1] (i = 1..3), c = UT spread, from packed L.
 */
__device__ __forceinline__ void
device_sigma_points_from_factor(const float mu[3], const float L[6], float c_spread, float pts_out[7][3])
{
    const float cols[3][3] = {{L[0], L[1], L[2]}, {0.f, L[3], L[4]}, {0.f, 0.f, L[5]}};
#pragma unroll
    for (int k = 0; k < 3; ++k)
        pts_out[0][k] = mu[k];
#pragma unroll
    for (int col = 0; col < 3; ++col)
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            pts_out[col + 1][k] = mu[k] + c_spread * cols[col][k];
            pts_out[col + 4][k] = mu[k] - c_spread * cols[col][k];
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exact tile box for an undistorted pinhole
// ─────────────────────────────────────────────────────────────────────────────

/** k² of the rasterizer's draw test: a ray is drawn iff its line passes within Mahalanobis k of the Gaussian. */
__device__ __forceinline__ float device_draw_k2(float opacity)
{
    return fminf(2.f * __logf(opacity / ALPHA_THRESHOLD), -2.f * __logf(MAX_RASTER_KERNEL_DENSITY));
}

/** Whether a camera is a pinhole with all distortion slots zero. */
template <ProjectionFamily F> __device__ __forceinline__ bool device_is_plain_pinhole(const float* dist)
{
    if constexpr (F != PINHOLE_FAMILY)
        return false;
#pragma unroll
    for (int i = 0; i < CameraOps<F>::n_slots; ++i)
        if (dist[i] != 0.f)
            return false;
    return true;
}

/** Hull of {u ∈ [lo, hi] : a u² − 2 b u + c ≤ 0}; false when the set is empty. */
__device__ __forceinline__ bool
device_quadratic_sublevel_hull(float a, float b, float c, float lo, float hi, float& out_lo, float& out_hi)
{
    // The set's boundary within [lo, hi] is made of the endpoints and the roots.
    out_lo = INFINITY;
    out_hi = -INFINITY;
    if ((a * lo - 2.f * b) * lo + c <= 0.f) {
        out_lo = lo;
        out_hi = lo;
    }
    if ((a * hi - 2.f * b) * hi + c <= 0.f) {
        out_lo = fminf(out_lo, hi);
        out_hi = hi;
    }
    const float disc = b * b - a * c;
    if (disc >= 0.f) {
        const float q = b + copysignf(sqrtf(disc), b); // stable root pair: q / a and c / q
        const float r1 = a != 0.f ? q / a : NAN;
        const float r2 = q != 0.f ? c / q : NAN;
        if (r1 >= lo && r1 <= hi) {
            out_lo = fminf(out_lo, r1);
            out_hi = fmaxf(out_hi, r1);
        }
        if (r2 >= lo && r2 <= hi) {
            out_lo = fminf(out_lo, r2);
            out_hi = fmaxf(out_hi, r2);
        }
    }
    return out_lo <= out_hi;
}

/**
 * Restrict [lo, hi] to the forward piece of {a u² − 2 b u + c ≤ 0} for an ellipse straddling z = 0 (a < 0).
 *
 * The set is then two half-lines: lines through the camera that meet the ellipse in front, and those that meet it
 * only behind. The forward one runs towards the side of the ellipse's z = 0 chord, sign(side). False: empty.
 */
__device__ __forceinline__ bool device_forward_piece(float a, float b, float c, float side, float& lo, float& hi)
{
    const float disc = b * b - a * c;
    if (a >= 0.f || disc < 0.f)
        return true; // in front of the camera entirely, or the camera's line meets it: every line hits the front
    const float q = b + copysignf(sqrtf(disc), b);
    const float r1 = q / a, r2 = c / q;
    if (side > 0.f)
        lo = fmaxf(lo, fmaxf(r1, r2));
    else
        hi = fminf(hi, fminf(r1, r2));
    return lo <= hi;
}

/**
 * Smallest image box holding every pixel whose ray the rasterizer draws for this Gaussian.
 *
 * The rasterizer draws a ray when D² = min_{t≥0} Mahalanobis² ≤ k², k² = min(2 ln(α/τ), −2 ln ρ_max).
 * The pixel column u is hit iff the plane through the camera and that column meets the forward part of the
 * k-ellipsoid; the plane meets the ellipsoid iff lᵀ(μμᵀ − k²Σ)l ≤ 0 for l = (1, 0, −u) in normalised coordinates
 * (the projected dual quadric); likewise for rows. For a Gaussian straddling the camera plane that set also holds
 * the planes meeting only its back part, which `device_forward_piece` removes. Camera-space μ, Σ in (μ_z > 0);
 * box clipped to the image.
 */
__device__ __forceinline__ bool device_exact_pinhole_box(
    const float mu_c[3],
    const float cov_c[6], // xx, xy, xz, yy, yz, zz
    float opacity,
    float fx,
    float fy,
    float cx,
    float cy,
    int image_width,
    int image_height,
    float& out_mean2d_x,
    float& out_mean2d_y,
    int32_t& out_radius_x,
    int32_t& out_radius_y
)
{
    const float k2 = device_draw_k2(opacity);
    const float a = mu_c[2] * mu_c[2] - k2 * cov_c[5];
    const float bx = mu_c[0] * mu_c[2] - k2 * cov_c[2];
    const float cx2 = mu_c[0] * mu_c[0] - k2 * cov_c[0];
    const float by = mu_c[1] * mu_c[2] - k2 * cov_c[4];
    const float cy2 = mu_c[1] * mu_c[1] - k2 * cov_c[3];
    // Side of each axis' z = 0 chord, scaled by Σ_zz > 0: x_c = μ_x − Σ_xz μ_z / Σ_zz.
    const float side_x = mu_c[0] * cov_c[5] - cov_c[2] * mu_c[2];
    const float side_y = mu_c[1] * cov_c[5] - cov_c[4] * mu_c[2];
    float ulo = -cx / fx, uhi = ((float)image_width - cx) / fx;
    float vlo = -cy / fy, vhi = ((float)image_height - cy) / fy;
    if (!device_forward_piece(a, bx, cx2, side_x, ulo, uhi) || !device_forward_piece(a, by, cy2, side_y, vlo, vhi))
        return false;
    float u0, u1, v0, v1;
    if (!device_quadratic_sublevel_hull(a, bx, cx2, ulo, uhi, u0, u1))
        return false;
    if (!device_quadratic_sublevel_hull(a, by, cy2, vlo, vhi, v0, v1))
        return false;
    const float x0 = fx * u0 + cx, x1 = fx * u1 + cx;
    const float y0 = fy * v0 + cy, y1 = fy * v1 + cy;
    out_mean2d_x = 0.5f * (x0 + x1);
    out_mean2d_y = 0.5f * (y0 + y1);
    // +1 px absorbs float error at the roots; isect_tiles widens the half-extent to whole tiles.
    out_radius_x = (int32_t)ceilf(0.5f * (x1 - x0)) + 1;
    out_radius_y = (int32_t)ceilf(0.5f * (y1 - y0)) + 1;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cone cap on the UT box for distorted lenses
// ─────────────────────────────────────────────────────────────────────────────

constexpr float PI_F = CameraOps<FISHEYE_FAMILY>::THETA_MAX; // π

/** Normalised box of the annular sector r ∈ [r0, r1], azimuth φ ± Δφ (Δφ >= π: full turn). */
__device__ __forceinline__ void
device_sector_box(float r0, float r1, float phi, float dphi, float& x0, float& x1, float& y0, float& y1)
{
    if (dphi >= PI_F) {
        x0 = -r1;
        x1 = r1;
        y0 = -r1;
        y1 = r1;
        return;
    }
    float s0, c0, s1, c1;
    sincosf(phi - dphi, &s0, &c0);
    sincosf(phi + dphi, &s1, &c1);
    x0 = fminf(fminf(r0 * c0, r0 * c1), fminf(r1 * c0, r1 * c1));
    x1 = fmaxf(fmaxf(r0 * c0, r0 * c1), fmaxf(r1 * c0, r1 * c1));
    y0 = fminf(fminf(r0 * s0, r0 * s1), fminf(r1 * s0, r1 * s1));
    y1 = fmaxf(fmaxf(r0 * s0, r0 * s1), fmaxf(r1 * s0, r1 * s1));
    // An axis direction inside the azimuth range reaches r1 along that axis.
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float psi = 0.5f * PI_F * (float)q;
        const float diff = remainderf(psi - phi, 2.f * PI_F);
        if (fabsf(diff) <= dphi) {
            x0 = fminf(x0, r1 * (q == 2 ? -1.f : 0.f));
            x1 = fmaxf(x1, r1 * (q == 0 ? 1.f : 0.f));
            y0 = fminf(y0, r1 * (q == 3 ? -1.f : 0.f));
            y1 = fmaxf(y1, r1 * (q == 1 ? 1.f : 0.f));
        }
    }
}

/**
 * Normalised box of the image of a circular cone of directions (axis at incidence θ_c, azimuth φ_c, half-angle β)
 * through a lens whose radius grows with incidence. False when the lens images none of it; `unbounded` when the
 * bound cannot be closed (keep the UT box). r_corner caps the radius: no pixel lies beyond it.
 */
template <ProjectionFamily F>
__device__ __forceinline__ bool device_cone_sector(
    float th_c,
    float ph_c,
    float beta,
    const float* dist,
    float r_corner,
    float& x0,
    float& x1,
    float& y0,
    float& y1,
    bool& unbounded
)
{
    const float th0 = fmaxf(0.f, th_c - beta);
    const float th1 = th_c + beta;
    float r0, r1;
    // The cull and clamp at r_corner ignore the non-radial offset, which could pull a point just past the corner
    // radius back onto the image; the offset bound at th1 is too loose to widen them by.
    if (!CameraOps<F>::radial(th0, dist, r0) || r0 > r_corner)
        return false;
    float mx = 0.f, my = 0.f;
    if (CameraOps<F>::radial(th1, dist, r1)) {
        r1 = fminf(r1, r_corner);
        CameraOps<F>::offset_bound(th1, dist, mx, my);
    }
    else if (CameraOps<F>::images_behind) {
        r1 = r_corner; // purely radial past its domain: the corner radius bounds every pixel
    }
    else {
        unbounded = true; // the non-radial offset has no bound past the domain
        return true;
    }
    const bool full = th_c <= beta || th_c + beta >= PI_F;
    const float dphi = full ? PI_F : asinf(fminf(1.f, sinf(beta) / sinf(th_c)));
    device_sector_box(r0, r1, ph_c, dphi, x0, x1, y0, y1);
    x0 -= mx;
    x1 += mx;
    y0 -= my;
    y1 += my;
    return true;
}

/**
 * Replace a UT box larger than the image (distorted lens) by the image of the cone of rays the rasterizer can draw.
 *
 * The rasterizer draws a ray when it passes within k of the Gaussian, so every drawn direction lies in the cone
 * around μ of half-angle asin(ρ/|μ|), with ρ² = k²·λmax(Σ) <= k²·‖Σ‖_F. Its image is an annular sector for any
 * lens whose radius grows with incidence.
 * A UT box that runs away in one axis has failed in both, so it is replaced, not intersected; where the cone cannot
 * be bounded the box is the whole image. False: cull.
 */
template <ProjectionFamily F>
__device__ __forceinline__ bool device_cone_cap(
    const float mu_c[3],
    const float cov_c[6],
    float opacity,
    float fx,
    float fy,
    float cx,
    float cy,
    const float* dist,
    int image_width,
    int image_height,
    float& mean2d_x,
    float& mean2d_y,
    int32_t& radius_x,
    int32_t& radius_y
)
{
    const float w = (float)image_width, h = (float)image_height;
    if (radius_x <= image_width && radius_y <= image_height)
        return true; // a box smaller than the image is the UT's own estimate; the cap targets the runaway ones
    const float k2 = device_draw_k2(opacity);
    const float frob = sqrtf(
        cov_c[0] * cov_c[0] + cov_c[3] * cov_c[3] + cov_c[5] * cov_c[5] +
        2.f * (cov_c[1] * cov_c[1] + cov_c[2] * cov_c[2] + cov_c[4] * cov_c[4])
    );
    const float rho2 = k2 * frob;
    const float d2 = mu_c[0] * mu_c[0] + mu_c[1] * mu_c[1] + mu_c[2] * mu_c[2];
    bool unbounded = d2 <= rho2; // camera inside the bounding sphere: every direction may be drawn
    bool any = false;
    float x0 = 0.f, x1 = 0.f, y0 = 0.f, y1 = 0.f;
    if (!unbounded) {
        const float beta = asinf(sqrtf(rho2 / d2));
        const float th = atan2f(sqrtf(mu_c[0] * mu_c[0] + mu_c[1] * mu_c[1]), mu_c[2]);
        const float ph = atan2f(mu_c[1], mu_c[0]);
        const float ux = fmaxf(cx, w - cx) / fx, uy = fmaxf(cy, h - cy) / fy;
        const float r_corner = sqrtf(ux * ux + uy * uy);
        any = device_cone_sector<F>(th, ph, beta, dist, r_corner, x0, x1, y0, y1, unbounded);
        if (!unbounded && !any)
            return false;
    }
    // Pixel box with 1 px margin, clipped to the image.
    const float nx0 = unbounded ? -1.f : fmaxf(fx * x0 + cx - 1.f, -1.f);
    const float nx1 = unbounded ? w + 1.f : fminf(fx * x1 + cx + 1.f, w + 1.f);
    const float ny0 = unbounded ? -1.f : fmaxf(fy * y0 + cy - 1.f, -1.f);
    const float ny1 = unbounded ? h + 1.f : fminf(fy * y1 + cy + 1.f, h + 1.f);
    if (nx1 <= nx0 || ny1 <= ny0)
        return false;
    mean2d_x = 0.5f * (nx0 + nx1);
    mean2d_y = 0.5f * (ny0 + ny1);
    radius_x = (int32_t)ceilf(0.5f * (nx1 - nx0)) + 1;
    radius_y = (int32_t)ceilf(0.5f * (ny1 - ny0)) + 1;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// UT projection — templated over the projection family (CameraOps<F>)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Project the 7 sigma points, reconstruct the 2D Gaussian and conic, and set the tile box (means2d centre, radii
 * half-extent) — a bound on the pixels the rasterizer draws, not a projection.
 *
 * Box: exact on an undistorted pinhole; otherwise the UT box (eps2d-dilated), replaced by the cone box when larger
 * than the image. Culled when the centre is outside [near, far], all points are behind the camera, opacity is below
 * threshold, the box misses the image, or det Σ₂d <= 0 for a UT-owned box. The conic is zero where det <= 0.
 * out_depth is the sort key ‖μ_c‖, not optical-axis depth.
 */
template <ProjectionFamily F>
__device__ __forceinline__ bool device_ut_project(
    const float pts[7][3],
    const float cov_L[6], // world factor L the sigma points were drawn from
    const float wm[7],
    const float wc[7],
    const float* __restrict__ viewmat,
    float fx,
    float fy,
    float cx,
    float cy,
    float eps2d,
    float opacity_cond,
    float near_plane,
    float far_plane,
    int image_width,
    int image_height,
    const float* dist,
    float& out_mean2d_x,
    float& out_mean2d_y,
    float& out_conic_a,
    float& out_conic_b,
    float& out_conic_c,
    float& out_depth,
    int32_t& out_radius_x,
    int32_t& out_radius_y
)
{
    const float r00 = viewmat[0], r01 = viewmat[1], r02 = viewmat[2];
    const float r10 = viewmat[4], r11 = viewmat[5], r12 = viewmat[6];
    const float r20 = viewmat[8], r21 = viewmat[9], r22 = viewmat[10];
    const float tx = viewmat[3], ty = viewmat[7], tz = viewmat[11];

    // Centre sigma point (σ₀ = Gaussian mean) in camera space.
    const float mc0x = r00 * pts[0][0] + r01 * pts[0][1] + r02 * pts[0][2] + tx;
    const float mc0y = r10 * pts[0][0] + r11 * pts[0][1] + r12 * pts[0][2] + ty;
    const float mc0z = r20 * pts[0][0] + r21 * pts[0][1] + r22 * pts[0][2] + tz;

    // Near/far cull on the optical-axis depth. Radial culling is only correct for lenses that
    // image rays with z ≤ 0; the families routed here (pinhole, Brown–Conrady, Kannala–Brandt)
    // are not among them. This is deliberately a different quantity from the sort key below —
    // culling decides which Gaussians exist, ordering decides how they composite.
    if (mc0z < near_plane || mc0z > far_plane)
        return false;

    // Tile sort key: ‖μ_c‖, the distance from the camera centre to the Gaussian mean — NOT
    // mc0z. The rasterizer intersects each pixel's ray with the 3D covariance, so the scalar
    // that orders two Gaussians along a shared ray is the distance along it; z is its
    // projection onto the optical axis and shrinks it by cos θ of the field angle. Gaussians
    // sharing a tile need not share a field angle, so a z key inverts pairs — the wider the
    // lens, the more of them, under an α-blend that does not commute.
    out_depth = sqrtf(mc0x * mc0x + mc0y * mc0y + mc0z * mc0z);

    // Project 7 sigma points world → camera → image
    float ip[7][2];
    bool any_valid = false;

    // Fallback image coords for sigma points the lens cannot image: the pinhole projection of the
    // centre, for every lens family.
    float fb_u = 0.f, fb_v = 0.f;
    CameraOps<PINHOLE_FAMILY>::project(mc0x, mc0y, mc0z, fx, fy, cx, cy, nullptr, fb_u, fb_v);

#pragma unroll
    for (int k = 0; k < 7; ++k) {
        const float wx = pts[k][0], wy = pts[k][1], wz = pts[k][2];
        const float ccx = r00 * wx + r01 * wy + r02 * wz + tx;
        const float ccy = r10 * wx + r11 * wy + r12 * wz + ty;
        const float ccz = r20 * wx + r21 * wy + r22 * wz + tz;

        float pu, pv;
        const bool ok = CameraOps<F>::project(ccx, ccy, ccz, fx, fy, cx, cy, dist, pu, pv);

        if (!ok) {
            ip[k][0] = fb_u;
            ip[k][1] = fb_v;
        }
        else {
            ip[k][0] = pu;
            ip[k][1] = pv;
            any_valid = true;
        }
    }

    if (!any_valid)
        return false;

    // UT reconstruction: 2D mean
    float mu2d_x = 0.f, mu2d_y = 0.f;
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        mu2d_x += wm[k] * ip[k][0];
        mu2d_y += wm[k] * ip[k][1];
    }

    // UT reconstruction: 2D covariance
    float c2d00 = 0.f, c2d01 = 0.f, c2d11 = 0.f;
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        const float dx = ip[k][0] - mu2d_x;
        const float dy = ip[k][1] - mu2d_y;
        c2d00 += wc[k] * dx * dx;
        c2d01 += wc[k] * dx * dy;
        c2d11 += wc[k] * dy * dy;
    }

    c2d00 += eps2d;
    c2d11 += eps2d;
    // det <= 0 culls only a box the UT owns: a runaway covariance cancels here, and its box is replaced below.
    const float det = c2d00 * c2d11 - c2d01 * c2d01;

    if (opacity_cond < ALPHA_THRESHOLD)
        return false;

    // Opacity-aware extent (StopThePop, arXiv:2402.00525 §B.2): a Gaussian only needs bounding
    // out to where its own alpha would still clear the draw threshold.
    const float extend = fminf(3.33f, sqrtf(2.f * __logf(opacity_cond / ALPHA_THRESHOLD)));
    // Per-axis half-extent of the 2D ellipse. Two guards that upstream gsplat carries here are
    // deliberately absent, both for the same reason: our UT covariance weights are all positive
    // (ut_constants.cuh), so Σ₂d is PSD by construction.
    //   · No negative-diagonal cull before the sqrt — a PSD diagonal cannot go negative, and
    //     the eps2d dilation above makes it strictly positive.
    //   · No min(·, extend·√v₁) clamp against the larger eigenvalue — for a PSD 2×2, v₁ is
    //     always ≥ either diagonal entry, so that bound is the bounding *circle* and never
    //     tightens the axis-aligned box.
    // Both guards exist upstream because a small-α scaled UT makes the centre weight negative
    // and Σ₂d indefinite. The static_assert on the weight signs is what keeps this valid.
    // Clamped before the cast: a runaway covariance exceeds int32, and out-of-range float→int is undefined in C++.
    out_radius_x = (int32_t)(ceilf(fminf(extend * sqrtf(c2d00), 1e8f)));
    out_radius_y = (int32_t)(ceilf(fminf(extend * sqrtf(c2d11), 1e8f)));

    out_mean2d_x = mu2d_x;
    out_mean2d_y = mu2d_y;

    // Undistorted pinhole: the exact box replaces the UT one, which is unbounded for Gaussians straddling the
    // camera plane and truncates others. Distorted lenses: the cone box replaces a UT box larger than the image.
    const bool plain = device_is_plain_pinhole<F>(dist);
    const bool runaway = out_radius_x > image_width || out_radius_y > image_height;
    if (!plain && !runaway && det <= 0.f)
        return false;
    // Camera-space μ, Σ for the replacement boxes; the viewmat rotation may carry the denorm scale, as μ_c does.
    const float mu_c[3] = {mc0x, mc0y, mc0z};
    float cov_c[6];
    if (plain || runaway) {
        const float R[3][3] = {{r00, r01, r02}, {r10, r11, r12}, {r20, r21, r22}};
        device_factor_congruence(R, cov_L, cov_c);
    }
    if (plain) {
        if (!device_exact_pinhole_box(
                mu_c,
                cov_c,
                opacity_cond,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                out_mean2d_x,
                out_mean2d_y,
                out_radius_x,
                out_radius_y
            ))
            return false;
    }
    else if (!device_cone_cap<F>(
                 mu_c,
                 cov_c,
                 opacity_cond,
                 fx,
                 fy,
                 cx,
                 cy,
                 dist,
                 image_width,
                 image_height,
                 out_mean2d_x,
                 out_mean2d_y,
                 out_radius_x,
                 out_radius_y
             ))
        return false;

    const float w = (float)image_width, h = (float)image_height;
    if (out_mean2d_x + out_radius_x <= 0.f || out_mean2d_x - out_radius_x >= w || out_mean2d_y + out_radius_y <= 0.f ||
        out_mean2d_y - out_radius_y >= h)
        return false;

    const float inv_det = det > 0.f ? 1.f / det : 0.f;
    out_conic_a = c2d11 * inv_det;
    out_conic_b = -c2d01 * inv_det;
    out_conic_c = c2d00 * inv_det;
    return true;
}
