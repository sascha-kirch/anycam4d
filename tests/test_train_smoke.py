"""``scripts/train.py:main`` trains from shipped configs for three steps, through Hydra argv.

Nothing about quality is asserted; every seam in the entry point must hold.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import anycam4d.utils
import numpy as np
import pytest
import torch
from plyfile import PlyData
from plyfile import PlyElement


_ROOT = Path(__file__).resolve().parents[1]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="train() builds a fused CUDA backend")


def _train_module():
    spec = importlib.util.spec_from_file_location("_train_script_smoke", _ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_point_cloud(data_dir: Path, n: int = 200) -> None:
    """Add the point cloud ``ModelFactory`` initializes from; the synthetic fixture has none."""
    ply = data_dir / "point_cloud.ply"
    if ply.exists():
        return
    rng = np.random.default_rng(0)
    pts = (rng.random((n, 3)) - 0.5).astype("f4")
    arr = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(ply))


def _run(tmp_path: Path, data_dir: Path, extra: list[str]) -> Path:
    """Invoke ``main`` through Hydra's argv path; group overrides in ``extra`` go first, as Hydra requires."""
    _ensure_point_cloud(data_dir)
    run_dir = tmp_path / "run"
    defaults = {
        "dataset.data_dir": str(data_dir),
        "hydra.run.dir": str(run_dir),
        "training.max_steps": "3",
        # Temporal tiles must divide the 2-frame fixture.
        "training.accum_renders": "2",
        "training.eval_every": "0",
        "training.save_every": "0",
        "training.log_every": "1",
        "model.sh_degree": "0",
        "strategy.cap_max": "2000",
        "strategy.refine_start_iter": "1",
        "strategy.refine_stop_iter": "3",
        "strategy.noise_injection_stop_iter": "2",
    }
    groups = [o for o in extra if "." not in o.split("=", 1)[0]]
    # The noise schedule is MCMC-only and the gradient-driven presets (the shipped default) do not declare it, so
    # under Hydra's struct mode overriding it there is an error rather than a no-op.
    if not any(g.startswith("recipe=") and "mcmc" in g for g in groups):
        del defaults["strategy.noise_injection_stop_iter"]
    for override in extra:
        key, _, value = override.partition("=")
        if "." in key:
            defaults[key] = value
    argv = [
        "train.py",
        # ``@hydra.main`` resolves its relative ``config_path`` against the module that defines it,
        # and importlib gives this one no package context — so point Hydra at ``conf/`` outright.
        "--config-path",
        str(_ROOT / "conf"),
        "--config-name",
        "4dgs",
        *groups,
        *[f"{k}={v}" for k, v in defaults.items()],
    ]
    saved = sys.argv
    sys.argv = argv
    try:
        _train_module().main()
    finally:
        sys.argv = saved
    return run_dir


@requires_cuda
@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("shipped", []),
        ("gradient_family", ["recipe=default_spatial"]),
        ("mcmc", ["recipe=mcmc_spatial"]),
        ("screenspace_trigger", ["strategy.spatial_trigger=screenspace"]),
        ("multiplicative_ewa_packed", ["model.appearance_model=sh4d_multiplicative"]),
        ("multiplicative_ewa_unpacked", ["model.appearance_model=sh4d_multiplicative", "rendering.packed=false"]),
        (
            "multiplicative_ut_packed",
            ["rendering=anycam4d", "model.appearance_model=sh4d_multiplicative"],
        ),
        (
            "multiplicative_ut_unpacked",
            ["rendering=anycam4d", "model.appearance_model=sh4d_multiplicative", "rendering.packed=false"],
        ),
        ("bilateral_grid_anycam4d", ["rendering=anycam4d", "rendering.post_processor=bilateral_grid"]),
        (
            "regularizer_variants",
            [
                "training.scale_reg_lambda=0.01",
                "training.opacity_reg_detach_gate=true",
                "training.opacity_reg_lambda=0.002",
            ],
        ),
    ],
)
def test_a_shipped_composition_trains(name: str, overrides: list[str], tmp_dataset: Path, tmp_path: Path) -> None:
    """Every shipped composition reaches the training loop.

    ``shipped`` (og_4dgs, param-space trigger) also checks provenance; ``gradient_family``, ``mcmc`` and
    ``screenspace_trigger`` take the other strategy branches; ``regularizer_variants`` threads the config-only
    regularizer options through ``main``.
    """
    run_dir = _run(tmp_path, tmp_dataset, overrides)

    assert (run_dir / "tb").is_dir(), "no TensorBoard directory: the writer never opened"
    assert list((run_dir / "tb").glob("events.out.tfevents.*")), "no event file: no step was logged"
    if name == "shipped":
        assert (run_dir / "run_provenance.json").is_file(), "the run recorded no code provenance"


@requires_cuda
def test_the_seed_reaches_the_rng_from_the_config(tmp_dataset: Path, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """``training.seed`` is applied before anything random is built (the call is asserted; runs are not bit-reproducible)."""
    seen: list[int] = []
    real_seed = anycam4d.utils.seed_everything
    # Patch the *source* binding, not a module instance: ``_run`` re-execs ``train.py`` for every
    # call, and its ``from anycam4d.utils import seed_everything`` rebinds at that moment.
    monkeypatch.setattr(anycam4d.utils, "seed_everything", lambda s: (seen.append(s), real_seed(s))[1])

    _run(tmp_path, tmp_dataset, ["training.seed=1234"])

    assert seen == [1234], f"seed_everything called {seen}, expected exactly [1234]"


@requires_cuda
def test_checkpointing_survives_the_whole_path(tmp_dataset: Path, tmp_path: Path) -> None:
    """A checkpoint written by the entry point carries what the read-only consumers need."""
    run_dir = _run(tmp_path, tmp_dataset, ["training.save_every=2"])

    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    assert checkpoints, "save_every was set and nothing was written"
    ckpt = torch.load(checkpoints[-1], weights_only=False)
    assert set(ckpt) >= {"model_state", "norm_scale", "norm_center", "config", "provenance"}


def test_the_default_schedule_refuses_a_three_step_run(tmp_dataset: Path, tmp_path: Path) -> None:
    """A shipped schedule that outlives the run stops it before a TensorBoard writer opens."""
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit):
        _run(tmp_path, tmp_dataset, ["strategy.refine_stop_iter=22500"])
    assert not (run_dir / "tb").exists(), "training started despite a schedule that outlives the run"
