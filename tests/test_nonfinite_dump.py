"""The non-finite dump names the offending primitives and keeps their values, gradients and Adam state."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from anycam4d.trainer.diagnostics import any_nonfinite
from anycam4d.trainer.diagnostics import dump_nonfinite
from anycam4d.trainer.diagnostics import nonfinite_rows


def test_dump_keeps_exactly_the_offending_rows(tmp_path: Path) -> None:
    n = 10
    params = {"means": nn.Parameter(torch.randn(n, 3)), "mu_t": nn.Parameter(torch.randn(n))}
    optimizers = {k: torch.optim.Adam([p]) for k, p in params.items()}
    for p in params.values():
        p.grad = torch.randn_like(p)
    for opt in optimizers.values():
        opt.step()

    assert not any_nonfinite(params)
    params["mu_t"].grad[3] = float("nan")  # a gradient goes bad first
    with torch.no_grad():
        params["means"][7, 1] = float("inf")  # a value already bad

    assert any_nonfinite(params)
    bad = nonfinite_rows(params)
    assert bad.nonzero().flatten().tolist() == [3, 7]

    path = tmp_path / "dump.pt"
    summary = dump_nonfinite(path, 42, params, optimizers, bad, {"timestamps": [0.1]})
    record = torch.load(path)
    assert record["ids"].tolist() == [3, 7]
    assert record["n_bad"] == 2
    exp_avg = optimizers["means"].state[params["means"]]["exp_avg"]
    torch.testing.assert_close(record["optimizer_state"]["means"]["exp_avg"], exp_avg[[3, 7]])
    assert torch.isnan(record["grads"]["mu_t"][0])
    assert "means: 1 non-finite values, 0 non-finite gradients" in summary
    assert "mu_t: 0 non-finite values, 1 non-finite gradients" in summary
