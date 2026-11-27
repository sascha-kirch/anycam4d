"""``scripts/eval.py``: JSON output, no images, CLI overrides, denormalization."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from anycam4d.config.dataset import DatasetConfig
from anycam4d.config.model import ModelConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.enums import AppearanceType
from anycam4d.enums import LpipsNet
from anycam4d.enums import PostProcessorType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


# Load scripts/eval.py as a module (scripts/ is not an importable package).
_EVAL_PATH = Path(__file__).parents[1] / "scripts" / "eval.py"
_spec = importlib.util.spec_from_file_location("eval_script", _EVAL_PATH)
assert _spec is not None and _spec.loader is not None
eval_script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eval_script  # so @dataclass can resolve the module for string annotations
_spec.loader.exec_module(eval_script)


def _section(obj: object) -> dict:
    """Resolved dict for a config dataclass, matching what the trainer persists."""
    return OmegaConf.to_container(OmegaConf.structured(obj), resolve=True)


def _config(data_dir: Path, *, post: PostProcessorType = PostProcessorType.NONE) -> dict:
    """A resolved config snapshot built from the real config dataclasses."""
    dataset = _section(DatasetConfig(data_dir=Path(data_dir), scale_factor=1))
    dataset["data_dir"] = str(data_dir)  # store as str (Hydra snapshots do); OmegaConf.create rejects Path
    return {
        "dataset": dataset,
        "rendering": _section(RenderingConfig(post_processor=post)),
        "model": _section(ModelConfig(sh_degree=0, appearance_model=AppearanceType.SH4D_ADDITIVE)),
        "training": _section(TrainingConfig(lpips_net=LpipsNet.ALEX)),
    }


def _make_checkpoint(
    data_dir: Path,
    *,
    post: PostProcessorType = PostProcessorType.NONE,
    with_config: bool = True,
    config_data_dir: Path | None = None,
) -> dict:
    """Minimal EWA checkpoint; ``config_data_dir`` lets the snapshot point away from ``data_dir``."""
    model = FourDGaussians.from_random(n_points=200, sh_degree=0, appearance_model=AppearanceType.SH4D_ADDITIVE)
    ckpt: dict = {
        "model_state": model.state_dict_with_meta(1.0, torch.zeros(3), 0.0, 1.0),
        "step": 1234,
        "norm_scale": 1.0,
        "norm_center": [0.0, 0.0, 0.0],
        "t_min": 0.0,
        "t_max": 1.0,
    }
    if with_config:
        ckpt["config"] = _config(config_data_dir or data_dir, post=post)
    return ckpt


def _write_ckpt(ckpt: dict, path: Path) -> Path:
    """Persist a checkpoint dict and return its path."""
    torch.save(ckpt, path)
    return path


def test_eval_script_smoke(tmp_dataset: Path, tmp_path: Path) -> None:
    """run_eval writes a JSON metrics file with the expected keys and finite metrics."""
    ckpt_path = _write_ckpt(_make_checkpoint(tmp_dataset), tmp_path / "step_001234.pt")
    out = tmp_path / "metrics.json"

    result = eval_script.run_eval(["--checkpoint", str(ckpt_path), "--output", str(out)])

    assert out.exists()
    payload = json.loads(out.read_text())
    assert set(payload) >= {
        "checkpoint",
        "step",
        "data_dir",
        "lpips_net",
        "mean",
        "std",
        "per_timestamp",
        "num_timesteps",
        "num_cameras",
        "num_images",
    }
    assert payload["step"] == 1234
    for key in ("psnr", "ssim", "lpips"):
        assert math.isfinite(result.mean[key])
        assert math.isfinite(payload["mean"][key])
    # No cc_* without a bilateral grid.
    assert "cc_psnr" not in payload["mean"]


def test_eval_writes_no_images(tmp_dataset: Path, tmp_path: Path) -> None:
    """No PNG (or any image) files are produced by the eval run."""
    ckpt_path = _write_ckpt(_make_checkpoint(tmp_dataset), tmp_path / "step_001234.pt")
    out_dir = tmp_path / "eval_out"

    eval_script.run_eval(["--checkpoint", str(ckpt_path), "--output", str(out_dir / "m.json")])

    assert list(out_dir.glob("*.png")) == []
    assert list(tmp_path.glob("**/*.png")) == []


def test_eval_missing_config_errors(tmp_dataset: Path, tmp_path: Path) -> None:
    """A checkpoint without a config snapshot raises SystemExit with a helpful message."""
    ckpt_path = _write_ckpt(_make_checkpoint(tmp_dataset, with_config=False), tmp_path / "step.pt")
    with pytest.raises(SystemExit, match="config snapshot"):
        eval_script.run_eval(["--checkpoint", str(ckpt_path)])


def test_eval_cli_overrides(tmp_dataset: Path, tmp_path: Path) -> None:
    """``--output`` redirects the JSON and ``--data_dir`` replaces a bogus stored path."""
    ckpt = _make_checkpoint(tmp_dataset, config_data_dir=Path("/nonexistent/seq"))
    norm = eval_script.compute_normalization(eval_script.MultiViewSequenceDataset(tmp_dataset, split="train"))
    ckpt["norm_scale"], ckpt["norm_center"] = norm.norm_scale, norm.norm_center.tolist()
    ckpt_path = _write_ckpt(ckpt, tmp_path / "step.pt")
    out = tmp_path / "custom" / "metrics.json"

    eval_script.run_eval(["--checkpoint", str(ckpt_path), "--output", str(out), "--data_dir", str(tmp_dataset)])

    assert out.exists()  # --output honored, default path not used
    payload = json.loads(out.read_text())
    assert payload["data_dir"] == str(tmp_dataset)  # --data_dir override won over the stored path


def test_eval_refuses_a_data_dir_of_another_capture(tmp_dataset: Path, tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_dataset)
    ckpt["norm_scale"], ckpt["norm_center"] = 3.0, [9.0, 9.0, 9.0]
    ckpt_path = _write_ckpt(ckpt, tmp_path / "step.pt")
    with pytest.raises(SystemExit, match="different captures"):
        eval_script.run_eval(["--checkpoint", str(ckpt_path), "--data_dir", str(tmp_dataset)])


def test_eval_applies_denorm(tmp_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset.apply_viewmat_transform is called with T_denorm reconstructed from norm_scale/center."""
    ckpt = _make_checkpoint(tmp_dataset)
    ckpt["norm_scale"] = 2.0
    ckpt["norm_center"] = [1.0, 2.0, 3.0]
    ckpt_path = _write_ckpt(ckpt, tmp_path / "step.pt")

    captured: dict[str, torch.Tensor] = {}
    real_ds = eval_script.MultiViewSequenceDataset

    def _spy(data_dir, split="train", scale_factor=1):  # noqa: ANN001, ANN202
        ds = real_ds(data_dir, split=split, scale_factor=scale_factor)
        orig = ds.apply_viewmat_transform

        def _wrap(t: torch.Tensor) -> None:
            captured["T"] = t.clone()
            orig(t)

        ds.apply_viewmat_transform = _wrap  # type: ignore[method-assign]
        return ds

    monkeypatch.setattr(eval_script, "MultiViewSequenceDataset", _spy)

    eval_script.run_eval(["--checkpoint", str(ckpt_path), "--output", str(tmp_path / "m.json")])

    expected = torch.eye(4)
    expected[:3, :3] *= 2.0
    expected[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    assert "T" in captured
    assert torch.allclose(captured["T"], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
