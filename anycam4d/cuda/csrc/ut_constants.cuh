/**
 * Unscented Transform constants (α, β, κ) = (1, 2, 0), fixed at compile time (scaled UT, D = 3):
 *   λ = α²(3 + κ) - 3 = 0,  c_spread = sqrt(3 + λ) = sqrt(3)
 *   W_m[0] = λ / (3 + λ) = 0,  W_m[i] = 0.5 / (3 + λ) = 1/6
 *   W_c[0] = W_m[0] + 1 - α² + β = 2,  W_c[i] = 1/6
 * c_spread uses a constexpr Newton square root; the static_asserts pin the values.
 */

#pragma once

// Compile-time Newton–Raphson square root (C++17 constexpr; host-evaluated, baked in).
constexpr float ct_sqrt(float x)
{
    if (x <= 0.f)
        return 0.f;
    float g = x;
    for (int i = 0; i < 30; ++i)
        g = 0.5f * (g + x / g);
    return g;
}

// ── Paper-frozen UT parameters ───────────────────────────────────────────────
constexpr float UT_ALPHA = 1.0f;
constexpr float UT_BETA = 2.0f;
constexpr float UT_KAPPA = 0.0f;

// ── Derived spread + weights (all compile-time) ──────────────────────────────
constexpr float UT_LAMBDA = UT_ALPHA * UT_ALPHA * (3.f + UT_KAPPA) - 3.f;
constexpr float UT_D_PLUS_LAMBDA = 3.f + UT_LAMBDA;
constexpr float UT_C_SPREAD = ct_sqrt(UT_D_PLUS_LAMBDA);
constexpr float UT_WM0 = UT_LAMBDA / UT_D_PLUS_LAMBDA;
constexpr float UT_WMI = 0.5f / UT_D_PLUS_LAMBDA;
constexpr float UT_WC0 = UT_WM0 + (1.f - UT_ALPHA * UT_ALPHA + UT_BETA);
constexpr float UT_WCI = UT_WMI;

// Pin the resolved values — break loudly if anyone edits α/β/κ without re-checking.
// Bounds are ~1e-6-wide (gross-drift guards, not bit-exactness — ct_sqrt may settle a
// ULP off the true float). Runtime bit-fidelity is gated by the kernel reference tests.
static_assert(UT_C_SPREAD > 1.732050f && UT_C_SPREAD < 1.732051f, "UT_C_SPREAD must be ≈ sqrt(3)");
static_assert(UT_WM0 == 0.f, "UT_WM0 must be 0 for (α,β,κ)=(1,2,0)");
static_assert(UT_WMI > 0.166666f && UT_WMI < 0.166667f, "UT_WMI must be ≈ 1/6");
static_assert(UT_WC0 == 2.f, "UT_WC0 must be 2 for (α,β,κ)=(1,2,0)");

// Every covariance weight is positive, so the reconstructed 2D covariance is PSD by construction:
// device_ut_project takes sqrt of its diagonal and needs no eigenvalue clamp. gsplat's α = 0.1 gives
// W_c[0] ≈ -96; if this assert fires, those guards must come back.
static_assert(
    UT_WC0 > 0.f && UT_WCI > 0.f,
    "UT covariance weights must be positive: device_ut_project assumes a PSD 2D covariance"
);
