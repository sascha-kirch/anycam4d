"""Temporal math operations for 4D Gaussian splatting.

Holds the temporal-variance epsilon, the one number shared by every expression that divides by
the marginal temporal variance. It lives in ``math/`` rather than beside the renderer because
both directions of use exist — ``math.covariance.schur_condition`` needs it and cannot import
from ``primitives/`` — and it is a plain float, so it costs the lazy-barrel contract in
``anycam4d/math/__init__.py`` nothing: importing this module pulls in ``math`` (stdlib) and
nothing else, which keeps it safe for the torch-free ``anycam4d/data/prep`` package.
"""

from __future__ import annotations

import math


#: Additive guard on the marginal temporal **variance**, in the temporal gate
#: ``w_t = exp(-0.5 * dt^2 / (cov_t_marg + eps))`` and in the Schur complement's denominator.
#:
#: It keeps a temporally collapsed primitive recoverable, which is a stronger job than avoiding a
#: division by zero: without it ``w_t`` reaches exactly zero at every timestamp the renderer visits,
#: so the primitive can no longer learn its way out and nothing prunes it. It is absolute, so what
#: it means depends on the sequence -- ``sqrt(eps)`` is 0.30 inter-frame ticks at 96 frames and 1.71
#: at 540 -- and it is additive, so a primitive whose own width equals the floor renders at
#: ``sqrt(2)`` floors. Below the floor the image loss supplies no counter-pressure, so any term that
#: charges for temporal width must charge for the *guarded* width. The collapse is demonstrated in
#: ``tests/test_temporal_floor_collapse.py``.
#:
#: **Invariant: this constant has exactly two definitions and they must move together.** Python
#: here, CUDA as ``TEMPORAL_VARIANCE_EPS`` in ``cuda/csrc/conditioning.cuh``. Editing one alone
#: desynchronises the fused path from the PyTorch oracle. Any new expression dividing by a temporal
#: variance imports this symbol rather than writing the literal.
TEMPORAL_VARIANCE_EPS = 1e-5

#: Narrowest temporal extent the renderer can distinguish, in units of the normalized span.
TEMPORAL_SIGMA_FLOOR = math.sqrt(TEMPORAL_VARIANCE_EPS)
