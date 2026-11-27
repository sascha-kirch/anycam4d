"""``TEMPORAL_VARIANCE_EPS`` has one definition per language, and the two agree.

Structural tests catch a re-typed literal without a GPU; the numerical test renders a collapsed
population, where the epsilon alone sets ``w_t``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from anycam4d.math.temporal import TEMPORAL_SIGMA_FLOOR
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS


_ROOT = Path(__file__).parents[1]
_CSRC = _ROOT / "anycam4d" / "cuda" / "csrc"
_DEFINITION_SITES = {
    Path("anycam4d/math/temporal.py"),
    Path("anycam4d/cuda/csrc/conditioning.cuh"),
}

# A line writing the literal beside a temporal-width name is a duplicate definition (``rtol=1e-5`` is not).
_TEMPORAL_NAME = re.compile(r"\b(cov_t_marg_safe|cov_t_marg|std_t_marg|cov_t|scale_t)\b")
_BARE_LITERAL = re.compile(r"(?<![\w.])1e-0*5f?\b")


def _sources() -> list[Path]:
    """Package Python and CUDA sources plus ``tests/oracles``, minus the two definition sites."""
    paths = sorted(_ROOT.joinpath("anycam4d").rglob("*.py"))
    paths += sorted(_ROOT.joinpath("tests", "oracles").rglob("*.py"))
    paths += sorted(p for ext in ("*.cu", "*.cuh", "*.h") for p in _CSRC.rglob(ext))
    return [p for p in paths if p.relative_to(_ROOT) not in _DEFINITION_SITES]


def test_no_source_re_types_the_temporal_epsilon() -> None:
    """No file outside the two definition sites writes ``1e-5`` beside a temporal variance."""
    offenders = []
    for path in _sources():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _BARE_LITERAL.search(line) and _TEMPORAL_NAME.search(line):
                offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "bare temporal epsilon — import TEMPORAL_VARIANCE_EPS from anycam4d.math.temporal "
        "(CUDA: it is #defined in conditioning.cuh, which every conditioning kernel includes):\n" + "\n".join(offenders)
    )


def test_cuda_defines_it_once_and_agrees_with_python() -> None:
    """Exactly one ``#define``, in the header every conditioning kernel includes, same value."""
    defines = [
        (path, line)
        for path in sorted(_CSRC.rglob("*.cu")) + sorted(_CSRC.rglob("*.cuh")) + sorted(_CSRC.rglob("*.h"))
        for line in path.read_text().splitlines()
        if line.startswith("#define TEMPORAL_VARIANCE_EPS")
    ]
    assert len(defines) == 1, f"expected one CUDA definition, found: {[str(p) for p, _ in defines]}"
    path, line = defines[0]
    assert path.name == "conditioning.cuh"
    assert float(line.split()[-1].rstrip("f")) == TEMPORAL_VARIANCE_EPS


def test_every_conditioning_kernel_can_see_the_definition() -> None:
    """A kernel that uses the symbol includes the header that defines it."""
    for path in sorted(_CSRC.rglob("*.cu")):
        text = path.read_text()
        if "TEMPORAL_VARIANCE_EPS" in text:
            assert '#include "conditioning.cuh"' in text, f"{path.name} uses the symbol without the include"


@pytest.mark.parametrize("dt_in_floors", [0.0, 1.0, 2.0])
def test_epsilon_dominated_gate_matches_the_fused_kernel(dt_in_floors: float) -> None:
    """Fused and oracle gates agree on a collapsed population: ``exp(-0.5 k²)`` at ``k`` floor-widths."""
    from anycam4d.cuda.ext import HAS_CUDA_EXT

    if not HAS_CUDA_EXT or not torch.cuda.is_available():
        pytest.skip("CUDA extension or device unavailable")

    from tests.fused_stages import condition
    from anycam4d.math.covariance import build_cov4d_rot

    n, tau, device = 512, 0.5, "cuda"
    g = torch.Generator(device=device).manual_seed(7)
    quats_l = torch.randn(n, 4, device=device, generator=g)
    quats_r = torch.randn(n, 4, device=device, generator=g)
    # exp(2 * -13.8) = 1e-12, i.e. cov_t_marg / eps = 1e-7 — the epsilon carries the denominator.
    scales_4d_log = torch.full((n, 4), -13.8, device=device)
    means = torch.randn(n, 3, device=device, generator=g)
    mu_t = torch.full((n,), tau - dt_in_floors * TEMPORAL_SIGMA_FLOOR, device=device)
    opacities = torch.zeros(n, device=device)

    cov_t_marg = build_cov4d_rot(quats_l, quats_r, scales_4d_log)[:, 3, 3]
    assert (cov_t_marg < TEMPORAL_VARIANCE_EPS * 1e-4).all(), "population is not epsilon-dominated"

    _, _, _, wt_fused = condition(quats_l, quats_r, scales_4d_log, means, mu_t, opacities, tau, 1e-6, 0.0)

    dt = tau - mu_t
    wt_oracle = torch.exp(-0.5 * dt * dt / (cov_t_marg + TEMPORAL_VARIANCE_EPS))
    torch.testing.assert_close(wt_fused, wt_oracle, rtol=1e-6, atol=1e-6)

    # The closed form the epsilon alone produces — pins the *value*, not just the agreement, so a
    # matched pair of wrong epsilons cannot pass.
    expected = pytest.approx(float(torch.exp(torch.tensor(-0.5 * dt_in_floors**2))), rel=1e-4)
    assert wt_fused.min().item() == expected
    assert wt_fused.max().item() == expected
