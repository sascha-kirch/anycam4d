"""Float32 conditioning benchmarks for two kernel expressions (manual, non-gating).

Each replicates a device expression exactly and scores it against float64 at a controlled
condition number. Only "the shipped form is not worse" is asserted.

- ``bench_covariance_route``: eigendecomposition vs Cramer inverse vs ``device_cholesky3x3``.
- ``bench_residual_projection``: scalar ``||y_p||² - b²/a`` vs residual projection on grazing rays.

Run: ``pixi run pytest tests/bench_numerics.py -m bench -s``.
"""

from __future__ import annotations

import time

import pytest
import torch


pytestmark = [
    pytest.mark.bench,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="benchmark requires CUDA"),
]

DEVICE = torch.device("cuda")
KAPPAS = (1e4, 1e6, 1e8)
# Grazing severity ||y_p||^2 / D^2 -- see _grazing_geometry for why this, not kappa, is the
# axis the residual-projection claim lives on, and why the values are the common operating range.
SEVERITIES = (1e4, 1e6, 1e8)
KAPPA_REF = 1e4  # a realistically thin conditioned Gaussian, for the severity sweep
N = 1 << 16  # 65536 — large enough for a stable median, small enough for the eigh route's workspace
TIMING_ITERS = 20
SEED = 0

# The diagonal clamp inside device_cholesky3x3; replicated so the float32 path here cannot be
# accidentally better conditioned than the kernel it stands in for.
CHOL_FLOOR = 1e-8

# cuSOLVER's batched eigensolver rejects large batches (~0.26 MB workspace per matrix).
EIGH_CHUNK = 1 << 12


def _random_rotations(n: int, generator: torch.Generator) -> torch.Tensor:
    """``[n, 3, 3]`` random rotations, from the QR factor of a Gaussian matrix (float64)."""
    a = torch.randn(n, 3, 3, generator=generator, device=DEVICE, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # Fix the sign convention so Q is uniformly distributed rather than QR-implementation defined.
    return q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)


def _spd_with_condition(n: int, kappa: float, generator: torch.Generator) -> torch.Tensor:
    """``[n, 3, 3]`` SPD matrices with condition number ``kappa``, eigenvalues log-spaced."""
    rot = _random_rotations(n, generator)
    eig = torch.tensor([1.0, kappa**-0.5, 1.0 / kappa], device=DEVICE, dtype=torch.float64)
    return rot @ torch.diag(eig) @ rot.transpose(-2, -1)


def _upper_tri(sigma: torch.Tensor) -> torch.Tensor:
    """``[n, 6]`` upper-triangular storage ``[s00, s01, s02, s11, s12, s22]`` (the kernel layout)."""
    return torch.stack(
        [sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 0, 2], sigma[:, 1, 1], sigma[:, 1, 2], sigma[:, 2, 2]],
        dim=-1,
    )


def _cholesky_packed(s: torch.Tensor) -> torch.Tensor:
    """``device_cholesky3x3`` verbatim; ``[n, 6]`` upper-tri covariance to packed lower-tri factor."""
    floor = torch.tensor(CHOL_FLOOR, dtype=s.dtype, device=s.device)
    l00 = torch.maximum(s[:, 0], floor).sqrt()
    l10 = s[:, 1] / l00
    l20 = s[:, 2] / l00
    l11 = torch.maximum(s[:, 3] - l10 * l10, floor).sqrt()
    l21 = (s[:, 4] - l20 * l10) / l11
    l22 = torch.maximum(s[:, 5] - l20 * l20 - l21 * l21, floor).sqrt()
    return torch.stack([l00, l10, l20, l11, l21, l22], dim=-1)


def _solve_lower(packed: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``device_solve_lower3x3`` verbatim: forward substitution against the packed factor."""
    y0 = b[:, 0] / packed[:, 0]
    y1 = (b[:, 1] - packed[:, 1] * y0) / packed[:, 3]
    y2 = (b[:, 2] - packed[:, 2] * y0 - packed[:, 4] * y1) / packed[:, 5]
    return torch.stack([y0, y1, y2], dim=-1)


def _mahal_cholesky(sigma_tri: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``D^2 = p^T Sigma^-1 p`` through the Cholesky factor + one forward substitution."""
    return _solve_lower(_cholesky_packed(sigma_tri), p).pow(2).sum(-1)


def _mahal_cramer(sigma: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``D²`` through an explicit cofactor inverse (squares the condition number)."""
    a, b, c = sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 0, 2]
    d, e = sigma[:, 1, 1], sigma[:, 1, 2]
    f = sigma[:, 2, 2]
    c00, c01, c02 = d * f - e * e, c * e - b * f, b * e - c * d
    det = a * c00 + b * c01 + c * c02
    c11, c12, c22 = a * f - c * c, b * c - a * e, a * d - b * b
    quad = (
        p[:, 0] * p[:, 0] * c00
        + p[:, 1] * p[:, 1] * c11
        + p[:, 2] * p[:, 2] * c22
        + 2.0 * (p[:, 0] * p[:, 1] * c01 + p[:, 0] * p[:, 2] * c02 + p[:, 1] * p[:, 2] * c12)
    )
    return quad / det


def _mahal_eigh(sigma: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``D²`` through ``eigh`` whitening, chunked because cuSOLVER rejects large batches."""
    out = torch.empty(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
    for lo in range(0, sigma.shape[0], EIGH_CHUNK):
        hi = min(lo + EIGH_CHUNK, sigma.shape[0])
        eigvals, eigvecs = torch.linalg.eigh(sigma[lo:hi])
        y = torch.einsum("nij,nj->ni", eigvecs.transpose(-2, -1), p[lo:hi])
        out[lo:hi] = (y * y / eigvals.clamp_min(CHOL_FLOOR)).sum(-1)
    return out


def _rel_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Median relative error against a float64 reference, over the finite entries only."""
    err = (value.double() - reference).abs() / reference.abs().clamp_min(1e-300)
    finite = err[torch.isfinite(err)]
    return float(finite.median()) if finite.numel() else float("nan")


def _nonfinite_rate(value: torch.Tensor) -> float:
    """Fraction of entries that came out NaN or infinite."""
    return float((~torch.isfinite(value)).double().mean())


def _time_us_per_item(fn, *args) -> float:
    """Microseconds per element for ``fn(*args)``, after a warmup, over ``TIMING_ITERS`` runs."""
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(TIMING_ITERS):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / TIMING_ITERS / N * 1e6


def _rule(title: str, header: str) -> None:
    print(f"\n{title}")
    print(header)
    print("-" * len(header))


def test_bench_covariance_route() -> None:
    """Cost, non-finite rate, and whitening accuracy of the three covariance routes."""
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    routes = ("eigh (R,S)", "Cramer inverse", "Cholesky (ours)")
    results: dict[str, dict[float, tuple[float, float]]] = {r: {} for r in routes}
    timings: dict[str, float] = {}

    for kappa in KAPPAS:
        sigma64 = _spd_with_condition(N, kappa, generator)
        p64 = torch.randn(N, 3, generator=generator, device=DEVICE, dtype=torch.float64)
        sigma32, p32 = sigma64.float(), p64.float()
        tri64, tri32 = _upper_tri(sigma64), _upper_tri(sigma32)

        reference = _mahal_cholesky(tri64, p64)  # float64, same expression
        values = {
            "eigh (R,S)": _mahal_eigh(sigma32, p32),
            "Cramer inverse": _mahal_cramer(sigma32, p32),
            "Cholesky (ours)": _mahal_cholesky(tri32, p32),
        }
        for name, value in values.items():
            results[name][kappa] = (_rel_error(value, reference), _nonfinite_rate(value))

        if kappa == KAPPAS[1]:  # time once, at the mid condition number
            timings["eigh (R,S)"] = _time_us_per_item(_mahal_eigh, sigma32, p32)
            timings["Cramer inverse"] = _time_us_per_item(_mahal_cramer, sigma32, p32)
            timings["Cholesky (ours)"] = _time_us_per_item(_mahal_cholesky, tri32, p32)

    header = f"{'route':<18}{'us/Gauss':>10}{'nonfinite':>12}" + "".join(f"{f'rel err k=1e{int(k):d}':>18}" for k in ())
    header = f"{'route':<18}{'us/Gauss':>10}{'nonfinite':>12}" + "".join(
        f"{'rel err ' + f'{k:.0e}':>18}" for k in KAPPAS
    )
    _rule(f"Covariance route (N={N}, float32 vs float64 reference)", header)
    for name in routes:
        worst_nonfinite = max(results[name][k][1] for k in KAPPAS)
        line = f"{name:<18}{timings[name]:>10.4f}{worst_nonfinite:>12.2e}"
        line += "".join(f"{results[name][k][0]:>18.3e}" for k in KAPPAS)
        print(line)

    mid = KAPPAS[1]
    assert results["Cholesky (ours)"][mid][0] <= results["Cramer inverse"][mid][0]
    assert results["Cholesky (ours)"][mid][1] <= results["eigh (R,S)"][mid][1] + 1e-12


def _grazing_geometry(
    kappa: float, severity: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rays grazing their Gaussian at ``severity = ||y_p||² / D²``, with ``D²`` known in closed form.

    Built in the whitened frame: ``y_d = u / ||L u||``, ``y_p = t·u + sqrt(D2)·w``. Severity is roughly
    ``(camera distance / Gaussian extent)²``, so 1e4-1e6 is the common case.

    Returns:
        ``(sigma [n,3,3], d_world [n,3], p_world [n,3], d2_true [n])`` in float64.
    """
    sigma = _spd_with_condition(N, kappa, generator)
    factor = torch.linalg.cholesky(sigma)

    u = torch.randn(N, 3, generator=generator, device=DEVICE, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    w = torch.randn(N, 3, generator=generator, device=DEVICE, dtype=torch.float64)
    w = w - (w * u).sum(-1, keepdim=True) * u
    w = w / w.norm(dim=-1, keepdim=True)

    d2_true = torch.ones(N, device=DEVICE, dtype=torch.float64)  # graze at one Mahalanobis sigma
    t = (d2_true * severity).sqrt().unsqueeze(-1)

    d_world = torch.einsum("nij,nj->ni", factor, u)
    d_world = d_world / d_world.norm(dim=-1, keepdim=True)
    p_world = torch.einsum("nij,nj->ni", factor, t * u + d2_true.sqrt().unsqueeze(-1) * w)
    return sigma, d_world, p_world, d2_true


def _d2_both_forms(sigma_tri: torch.Tensor, d: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(scalar, residual)`` squared Mahalanobis distance, both from the same whitened vectors."""
    packed = _cholesky_packed(sigma_tri)
    y_d = _solve_lower(packed, d)
    y_p = _solve_lower(packed, p)
    a = y_d.pow(2).sum(-1)
    b = (y_d * y_p).sum(-1)
    scalar = y_p.pow(2).sum(-1) - b * b / a
    residual = (y_p - (b / a).unsqueeze(-1) * y_d).pow(2).sum(-1)
    return scalar, residual


def _residual_cell(kappa: float, severity: float, generator: torch.Generator) -> tuple[float, float]:
    """``(scalar rel err, residual rel err)`` at one ``(kappa, severity)`` point."""
    sigma, d, p, d2_true = _grazing_geometry(kappa, severity, generator)
    scalar, residual = _d2_both_forms(_upper_tri(sigma.float()), d.float(), p.float())
    return _rel_error(scalar, d2_true), _rel_error(residual, d2_true)


def test_bench_residual_projection() -> None:
    """``D^2`` error of the scalar form vs residual projection, over grazing severity and kappa."""
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    header = f"{'form':<20}" + "".join(f"{'sev=' + f'{s:.0e}':>16}" for s in SEVERITIES)
    _rule(f"Grazing-ray D^2 relative error at kappa={KAPPA_REF:.0e} (N={N}, float32)", header)
    cells = {s: _residual_cell(KAPPA_REF, s, generator) for s in SEVERITIES}
    print(f"{'scalar c - b^2/a':<20}" + "".join(f"{cells[s][0]:>16.3e}" for s in SEVERITIES))
    print(f"{'residual (ours)':<20}" + "".join(f"{cells[s][1]:>16.3e}" for s in SEVERITIES))
    print(f"{'improvement':<20}" + "".join(f"{cells[s][0] / max(cells[s][1], 1e-300):>15.0f}x" for s in SEVERITIES))

    # The scalar form's error is set by the cancellation alone; the residual form's is set by the
    # accuracy of the whitening that precedes it, so a badly conditioned Sigma caps the gain.
    header = f"{'kappa':<20}" + "".join(f"{'sev=' + f'{s:.0e}':>16}" for s in SEVERITIES)
    _rule("Improvement factor over (kappa, severity)", header)
    for kappa in KAPPAS:
        row = [_residual_cell(kappa, s, generator) for s in SEVERITIES]
        print(f"{kappa:<20.0e}" + "".join(f"{c[0] / max(c[1], 1e-300):>15.0f}x" for c in row))

    for severity in SEVERITIES:
        assert cells[severity][1] <= cells[severity][0]
