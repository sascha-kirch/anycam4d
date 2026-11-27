/**
 * Per-family projection math behind a compile-time tag: template<ProjectionFamily F> struct CameraOps.
 * project (camera xyz -> pixel) serves the UT sigma points; unproject (normalized pixel -> ray) serves
 * the world rasterizer. No Jacobians: gradients reach the Gaussians through the UT moment match and
 * the Mahalanobis backward. n_slots is the family's fixed distortion width (12 pinhole, 4 fisheye).
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "common.h"

template <ProjectionFamily F> struct CameraOps; // primary template — undefined

// ─────────────────────────────────────────────────────────────────────────────
// PINHOLE_FAMILY — perspective divide + OpenCV Brown–Conrady distortion
// ─────────────────────────────────────────────────────────────────────────────

template <> struct CameraOps<PINHOLE_FAMILY> {
    // 12-slot canonical OpenCV layout [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4].
    static constexpr int n_slots = 12;

    // Newton iteration budget for the inverse (undistort) solve. gsplat caps this at 5
    // (``N_MAX_UNDISTORTION_ITERATIONS``), which on a wide, strongly distorted lens returns an
    // unconverged iterate — i.e. a *wrong rendered ray*, not merely a slow one. The loop's
    // residual early-exit means the cap binds only on the strongly distorted periphery: typical
    // pixels leave after 2-3 iterations, and a zero-distortion pixel after exactly one (the
    // Newton step is identically zero there, so the undistorted path stays bit-identical at any
    // cap). Raising the cap from 5 to 20 costs no measurable render time.
    static constexpr int N_NEWTON = 20;

    /**
     * Project through OpenCV Brown-Conrady (dist = [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]).
     *
     * Returns false behind the camera or outside the lens's monotone radial domain, the same condition
     * unproject and the host footprint use; the sigma point is then dropped. gsplat's icD > 0.8 guard is
     * not used: it rejects points strongly distorted but still injective lenses image correctly.
     */
    static __device__ __forceinline__ bool project(
        float cx_cam,
        float cy_cam,
        float cz_cam,
        float fx,
        float fy,
        float cx_px,
        float cy_px,
        const float* d, // [12] canonical OpenCV layout [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
        float& out_u,
        float& out_v
    )
    {
        if (cz_cam <= 0.f)
            return false;
        const float inv_z = 1.f / cz_cam;
        // ``d == nullptr`` is the undistorted-pinhole centre fallback (device_ut_project,
        // ut_4d.cuh): a behind-camera sigma point's plain perspective projection. Treat as
        // zero distortion — never dereference null. Same grouping as the distorted path below, so
        // the two agree bit for bit at zero distortion.
        if (d == nullptr) {
            out_u = fx * cx_cam * inv_z + cx_px;
            out_v = fy * cy_cam * inv_z + cy_px;
            return true;
        }
        const float u = cx_cam * inv_z, v = cy_cam * inv_z; // normalised camera coords
        const float r2 = u * u + v * v;
        const float num = 1.f + r2 * (d[0] + r2 * (d[1] + r2 * d[4])); // k1,k2,k3
        const float den = 1.f + r2 * (d[5] + r2 * (d[6] + r2 * d[7])); // k4,k5,k6 (==1 when zero)
        // Monotone-domain test at r², identical in form to the one unproject() applies at its
        // solved radius: with s = r², d r_d/dr ∝ num·den + 2s(num'·den − num·den'), and a
        // non-positive den is a pole that ends the domain just the same. Zero distortion gives
        // num = den = 1 and num' = den' = 0, so the test is trivially true and the projection
        // below stays bit-identical to the bare perspective divide.
        const float num_s = d[0] + r2 * (2.f * d[1] + r2 * 3.f * d[4]); // num'(s)
        const float den_s = d[5] + r2 * (2.f * d[6] + r2 * 3.f * d[7]); // den'(s)
        if (den <= 1e-8f || (num * den + 2.f * r2 * (num_s * den - num * den_s)) <= 0.f)
            return false; // outside the lens's physical domain — invalid, not NaN
        const float icD = num / den;
        const float a1 = 2.f * u * v, a2 = r2 + 2.f * u * u, a3 = r2 + 2.f * v * v;
        const float dx = d[2] * a1 + d[3] * a2 + r2 * (d[8] + r2 * d[9]);   // p1,p2,s1,s2
        const float dy = d[2] * a3 + d[3] * a1 + r2 * (d[10] + r2 * d[11]); // p1,p2,s3,s4
        // Grouped so zero distortion (icD=1, dx=dy=0) is bit-identical to the plain divide
        // ``fx*cx_cam*inv_z + cx_px``, so the packed and dense UT kernels agree at visibility boundaries.
        out_u = fx * cx_cam * inv_z * icD + fx * dx + cx_px;
        out_v = fy * cy_cam * inv_z * icD + fy * dy + cy_px;
        return true;
    }

    /**
     * Ray (x, y, 1) for a Brown-Conrady pixel by analytic Newton undistort (gsplat Cameras.cuh), from
     * normalized distorted coordinates. Guards keep the last finite iterate.
     *
     * Returns false where the solved radius leaves the monotone domain: with s = r²,
     *   d r_d/dr ∝ num·den + 2s (num'·den - num·den') > 0, and den > 0.
     * Zero distortion is always valid and bit-identical to the plain perspective ray.
     */
    static __device__ __forceinline__ bool unproject(
        float u_n,
        float v_n,
        const float* d, // [12] canonical OpenCV layout [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
        float& out_dx,
        float& out_dy,
        float& out_dz
    )
    {
        // ``d == nullptr`` (or zero distortion) → trivial undistorted ray (u_n, v_n, 1).
        if (d == nullptr) {
            out_dx = u_n;
            out_dy = v_n;
            out_dz = 1.f;
            return true;
        }
        const float xd = u_n, yd = v_n; // distorted normalised coords (gsplat xd,yd)
        float x = u_n, y = v_n;         // initial guess
        constexpr float eps = 1e-6f;
        for (int it = 0; it < N_NEWTON; ++it) {
            const float r = x * x + y * y;
            const float r2 = r * r;
            const float alpha = 1.f + r * (d[0] + r * (d[1] + r * d[4])); // k1,k2,k3
            const float beta = 1.f + r * (d[5] + r * (d[6] + r * d[7]));  // k4,k5,k6
            const float dd = alpha / beta;                                // icD
            if (dd <= 0.f)                                                // flip guard — keep best (x,y)
                break;
            const float fx_res = dd * x + 2.f * d[2] * x * y + d[3] * (r + 2.f * x * x) + d[8] * r + d[9] * r2 - xd;
            const float fy_res = dd * y + 2.f * d[3] * x * y + d[2] * (r + 2.f * y * y) + d[10] * r + d[11] * r2 - yd;
            const float alpha_r = d[0] + r * (2.f * d[1] + r * 3.f * d[4]);
            const float beta_r = d[5] + r * (2.f * d[6] + r * 3.f * d[7]);
            const float d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta);
            const float d_x = 2.f * x * d_r;
            const float d_y = 2.f * y * d_r;
            float fx_x = dd + d_x * x + 2.f * d[2] * y + 6.f * d[3] * x;
            fx_x += 2.f * x * (d[8] + 2.f * d[9] * r);
            float fx_y = d_y * x + 2.f * d[2] * x + 2.f * d[3] * y;
            fx_y += 2.f * y * (d[8] + 2.f * d[9] * r);
            float fy_x = d_x * y + 2.f * d[3] * y + 2.f * d[2] * x;
            fy_x += 2.f * x * (d[10] + 2.f * d[11] * r);
            float fy_y = dd + d_y * y + 2.f * d[3] * x + 6.f * d[2] * y;
            fy_y += 2.f * y * (d[10] + 2.f * d[11] * r);
            const float det = fx_y * fy_x - fx_x * fy_y;
            if (fabsf(det) < eps) // singular Jacobian guard
                break;
            const float ddx = (fx_res * fy_y - fy_res * fx_y) / det;
            const float ddy = (fy_res * fx_x - fx_res * fy_x) / det;
            x += ddx;
            y += ddy;
            if (fabsf(ddx) < eps && fabsf(ddy) < eps) // residual early-exit
                break;
        }
        out_dx = x;
        out_dy = y;
        out_dz = 1.f;

        // Monotone-domain test at the solved radius — see the @return note above.
        const float s = x * x + y * y;
        const float n = 1.f + s * (d[0] + s * (d[1] + s * d[4]));   // num(s): k1,k2,k3
        const float m = 1.f + s * (d[5] + s * (d[6] + s * d[7]));   // den(s): k4,k5,k6
        const float n_s = d[0] + s * (2.f * d[1] + s * 3.f * d[4]); // num'(s)
        const float m_s = d[5] + s * (2.f * d[6] + s * 3.f * d[7]); // den'(s)
        return m > 0.f && (n * m + 2.f * s * (n_s * m - n * m_s)) > 0.f;
    }

    // Brown–Conrady images only z > 0: a direction past the camera plane is never on a pixel.
    static constexpr bool images_behind = false;

    /** Normalised radius of the radial image of incidence θ; false outside the monotone domain or past π/2. */
    static __device__ __forceinline__ bool radial(float theta, const float* d, float& r_out)
    {
        if (theta >= 1.5707f)
            return false;
        const float ru = tanf(theta);
        const float s = ru * ru;
        const float n = 1.f + s * (d[0] + s * (d[1] + s * d[4]));
        const float m = 1.f + s * (d[5] + s * (d[6] + s * d[7]));
        const float n_s = d[0] + s * (2.f * d[1] + s * 3.f * d[4]);
        const float m_s = d[5] + s * (2.f * d[6] + s * 3.f * d[7]);
        if (!(m > 1e-8f && (n * m + 2.f * s * (n_s * m - n * m_s)) > 0.f))
            return false;
        r_out = ru * n / m;
        return true;
    }

    /** Per-axis bound on the tangential and thin-prism offset for incidence ≤ θ (θ inside the domain). */
    static __device__ __forceinline__ void offset_bound(float theta, const float* d, float& mx, float& my)
    {
        const float ru = tanf(theta);
        const float s = ru * ru; // |2uv| <= s, s + 2u² <= 3s
        mx = s * (fabsf(d[2]) + 3.f * fabsf(d[3]) + fabsf(d[8]) + fabsf(d[9]) * s);
        my = s * (3.f * fabsf(d[2]) + fabsf(d[3]) + fabsf(d[10]) + fabsf(d[11]) * s);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// FISHEYE_FAMILY — equidistant (OpenCV) fisheye, KB θ-polynomial k1..k4
// ─────────────────────────────────────────────────────────────────────────────

template <> struct CameraOps<FISHEYE_FAMILY> {
    // 4-slot fisheye radial distortion [k1,k2,k3,k4].
    static constexpr int n_slots = 4;

    // Newton iteration budget for the θd → θ inverse. Same reasoning (and same measured cost,
    // i.e. none) as the pinhole family's ``N_NEWTON``: an unconverged iterate is a wrong
    // rendered ray, and the residual early-exit means the cap binds only near the fold.
    static constexpr int N_NEWTON = 20;

    // Hard upper bound on the incidence angle a pixel may invert to. θ = π is the ray
    // arriving from directly behind the camera — a full half turn off the optical axis, so
    // the whole sphere is already covered below it and a solve landing beyond has left the
    // model's domain. Deliberately NOT the hemisphere bound π/2: a real fisheye images past
    // a 180° FoV, and gating those pixels would be stricter than the host footprint.
    static constexpr float THETA_MAX = 3.14159265358979323846f;

    /**
     * Project through a Kannala-Brandt fisheye: θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸), θ = atan2(r, z).
     * Returns false behind the camera plane.
     *
     * Known gap: unproject admits angles up to π, so pixels past 90 degrees get rays whose geometry is
     * never tiled. Lifting this gate made training worse; see the strict xfails in test_camera_ops_distortion.py.
     */
    static __device__ __forceinline__ bool project(
        float cx_cam,
        float cy_cam,
        float cz_cam,
        float fx,
        float fy,
        float cx_px,
        float cy_px,
        const float* dist, // [4]: k1..k4
        float& out_u,
        float& out_v
    )
    {
        if (cz_cam <= 0.f)
            return false;
        const float k1 = dist[0], k2 = dist[1], k3 = dist[2], k4 = dist[3];
        const float r = sqrtf(cx_cam * cx_cam + cy_cam * cy_cam);
        const float theta = atan2f(r, cz_cam);
        const float theta2 = theta * theta;
        const float theta4 = theta2 * theta2;
        const float theta_d = theta * (1.f + k1 * theta2 + k2 * theta4 + k3 * theta4 * theta2 + k4 * theta4 * theta4);
        const float scale = (r > 1e-6f) ? (theta_d / r) : 1.f;
        out_u = fx * scale * cx_cam + cx_px;
        out_v = fy * scale * cy_cam + cy_px;
        return true;
    }

    /**
     * Unit ray for a fisheye pixel, inverting θd(θ) with five Newton steps.
     * Returns false outside θ ∈ [0, THETA_MAX] or past the fold dθd/dθ <= 0, as the host footprint does.
     */
    static __device__ __forceinline__ bool unproject(
        float u_n,
        float v_n,
        const float* dist, // [4]: k1..k4
        float& out_dx,
        float& out_dy,
        float& out_dz
    )
    {
        const float k1 = dist[0], k2 = dist[1], k3 = dist[2], k4 = dist[3];
        const float r_d = sqrtf(u_n * u_n + v_n * v_n); // distorted radial distance

        // Undistort: solve  theta * (1 + k1*t2 + k2*t4 + k3*t6 + k4*t8) = r_d  for theta
        float theta = r_d; // initial guess
        for (int i = 0; i < N_NEWTON; ++i) {
            const float t2 = theta * theta;
            const float t4 = t2 * t2;
            const float t6 = t4 * t2;
            const float t8 = t4 * t4;
            const float k = 1.f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8;
            const float dk = 2.f * theta * (k1 + 2.f * k2 * t2 + 3.f * k3 * t4 + 4.f * k4 * t6);
            const float f = theta * k - r_d;
            const float df = k + theta * dk;
            const float step = f / (df + 1e-10f);
            theta -= step;
            if (fabsf(step) < 1e-6f) // residual early-exit, as in the pinhole solve
                break;
        }

        // Ray direction from undistorted angle
        const float sin_theta = sinf(theta);
        const float cos_theta = cosf(theta);
        const float scale = (r_d > 1e-6f) ? (sin_theta / r_d) : 1.f;
        out_dx = scale * u_n;
        out_dy = scale * v_n;
        out_dz = cos_theta;

        // The solved angle must be a physical incidence angle: inside the half-turn range and
        // on the monotone side of the θ-polynomial's first turning point. Evaluated after the
        // ray is written — a gated pixel still carries a finite ray, the caller decides to
        // skip it.
        const float t2 = theta * theta;
        const float t4 = t2 * t2;
        const float dtheta_d = 1.f + 3.f * k1 * t2 + 5.f * k2 * t4 + 7.f * k3 * t4 * t2 + 9.f * k4 * t4 * t4;
        return theta >= 0.f && theta <= THETA_MAX && dtheta_d > 0.f;
    }

    // A fisheye past 180° images directions behind the camera plane.
    static constexpr bool images_behind = true;

    /** Normalised radius θd(θ) of incidence θ; false past π or the fold dθd/dθ <= 0. */
    static __device__ __forceinline__ bool radial(float theta, const float* d, float& r_out)
    {
        if (theta >= THETA_MAX)
            return false;
        const float t2 = theta * theta;
        const float slope = 1.f + t2 * (3.f * d[0] + t2 * (5.f * d[1] + t2 * (7.f * d[2] + t2 * 9.f * d[3])));
        if (slope <= 0.f)
            return false;
        r_out = theta * (1.f + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))));
        return true;
    }

    /** Kannala–Brandt is purely radial. */
    static __device__ __forceinline__ void offset_bound(float, const float*, float& mx, float& my)
    {
        mx = 0.f;
        my = 0.f;
    }
};
