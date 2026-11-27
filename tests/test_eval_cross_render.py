"""Scoring a checkpoint through a backend it was not trained with, and the normalization guard that makes it valid."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from anycam4d.config.dataset import DatasetConfig
from anycam4d.config.model import ModelConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization
from anycam4d.enums import AppearanceType
from anycam4d.enums import BackendType
from anycam4d.enums import LpipsNet
from anycam4d.enums import PostProcessorType
from anycam4d.eval.checkpoint import build_pipeline_from_checkpoint
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


_EVAL_PATH = Path(__file__).parents[1] / "scripts" / "eval.py"
_spec = importlib.util.spec_from_file_location("eval_script_cross", _EVAL_PATH)
assert _spec is not None and _spec.loader is not None
eval_script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eval_script
_spec.loader.exec_module(eval_script)


def _section(obj: object) -> dict:
    """Resolved dict for a config dataclass, matching what the trainer persists."""
    return OmegaConf.to_container(OmegaConf.structured(obj), resolve=True)


def _stored(backend: BackendType = BackendType.EWA, *, post: PostProcessorType = PostProcessorType.NONE) -> dict:
    """A stored ``rendering`` block, as a checkpoint's config snapshot carries it."""
    return _section(RenderingConfig(backend=backend, post_processor=post))


def _checkpoint(data_dir: Path, *, norm_scale: float, norm_center: list[float]) -> dict:
    """A minimal EWA checkpoint pinned to a caller-chosen scene normalization."""
    model = FourDGaussians.from_random(n_points=200, sh_degree=0, appearance_model=AppearanceType.SH4D_ADDITIVE)
    dataset = _section(DatasetConfig(data_dir=Path(data_dir), scale_factor=1))
    dataset["data_dir"] = str(data_dir)
    return {
        "model_state": model.state_dict_with_meta(norm_scale, torch.tensor(norm_center), 0.0, 1.0),
        "step": 1234,
        "norm_scale": norm_scale,
        "norm_center": norm_center,
        "t_min": 0.0,
        "t_max": 1.0,
        "config": {
            "dataset": dataset,
            "rendering": _stored(),
            "model": _section(ModelConfig(sh_degree=0, appearance_model=AppearanceType.SH4D_ADDITIVE)),
            "training": _section(TrainingConfig(lpips_net=LpipsNet.ALEX)),
        },
    }


def _dataset_normalization(data_dir: Path) -> tuple[float, list[float]]:
    """The ``(scale, centre)`` a run on ``data_dir`` would have trained at."""
    norm = compute_normalization(MultiViewSequenceDataset(data_dir, split="train", scale_factor=1))
    return norm.norm_scale, norm.norm_center.tolist()


def _aligned_checkpoint(data_dir: Path, tmp_path: Path) -> Path:
    """A checkpoint whose normalization matches ``data_dir``, persisted; returns its path."""
    scale, centre = _dataset_normalization(data_dir)
    path = tmp_path / "step_001234.pt"
    torch.save(_checkpoint(data_dir, norm_scale=scale, norm_center=centre), path)
    return path


def test_no_override_is_the_stored_config() -> None:
    """With neither flag the checkpoint's own block is used and nothing is flagged as changed."""
    cfg, is_override = eval_script.resolve_rendering(_stored(BackendType.ANYCAM4D), None, [])
    assert cfg.backend is BackendType.ANYCAM4D
    assert is_override is False


def test_preset_replaces_the_stored_config() -> None:
    """A preset name swaps the whole block, not just the keys it happens to mention."""
    cfg, is_override = eval_script.resolve_rendering(_stored(BackendType.EWA), "anycam4d", [])
    assert cfg.backend is BackendType.ANYCAM4D
    assert is_override is True


def test_set_patches_on_top_of_the_preset() -> None:
    """``--set`` is applied after the preset; a preset that also sets the key must not win."""
    cfg, _ = eval_script.resolve_rendering(_stored(), "anycam4d", ["packed=false"])
    assert cfg.backend is BackendType.ANYCAM4D
    assert cfg.packed is False  # anycam4d.yaml ships packed: true


def test_set_alone_patches_the_stored_config() -> None:
    """``--set`` without a preset patches the stored block and counts as a cross-render."""
    cfg, is_override = eval_script.resolve_rendering(_stored(BackendType.ANYCAM4D), None, ["eps2d=0.5"])
    assert cfg.backend is BackendType.ANYCAM4D
    assert cfg.eps2d == pytest.approx(0.5)
    assert is_override is True


def test_unknown_preset_names_the_options() -> None:
    """A typo'd preset fails loud and lists what exists, rather than silently keeping the stored one."""
    with pytest.raises(SystemExit, match="ewa"):
        eval_script.resolve_rendering(_stored(), "ewa_fusedd", [])


def test_override_reaches_the_backend(tmp_dataset: Path, tmp_path: Path) -> None:
    """The rebuilt pipeline holds the overridden backend, not the one the checkpoint stored."""
    ckpt = _checkpoint(tmp_dataset, norm_scale=1.0, norm_center=[0.0, 0.0, 0.0])
    device = torch.device("cpu")

    _, stored_pipeline = build_pipeline_from_checkpoint(ckpt, device)
    override = RenderingConfig.from_omegaconf(OmegaConf.create(_stored(BackendType.ANYCAM4D)))
    _, crossed_pipeline = build_pipeline_from_checkpoint(ckpt, device, rendering_override=override)

    assert type(stored_pipeline.backend) is not type(crossed_pipeline.backend)
    assert "UT" in type(crossed_pipeline.backend).__name__


def test_mismatched_normalization_refuses(tmp_dataset: Path, tmp_path: Path) -> None:
    """A fit from another capture is refused, not rendered outside every frustum."""
    path = tmp_path / "step.pt"
    torch.save(_checkpoint(tmp_dataset, norm_scale=999.0, norm_center=[7.0, 7.0, 7.0]), path)

    with pytest.raises(SystemExit, match="different captures"):
        eval_script.run_eval(
            ["--checkpoint", str(path), "--rendering", "anycam4d", "--output", str(tmp_path / "m.json")]
        )


def test_matching_normalization_runs(tmp_dataset: Path, tmp_path: Path) -> None:
    """A checkpoint in the dataset's own space cross-renders end to end and records that it did."""
    path = _aligned_checkpoint(tmp_dataset, tmp_path)
    out = tmp_path / "m.json"

    eval_script.run_eval(["--checkpoint", str(path), "--rendering", "anycam4d", "--output", str(out)])

    payload = json.loads(out.read_text())
    assert payload["cross_render"] is True
    assert payload["backend"] == BackendType.ANYCAM4D.value


def test_cc_follows_the_run_not_the_override(tmp_dataset: Path, tmp_path: Path) -> None:
    """``cc_*`` metrics follow whether the run trained a bilateral grid, not the override preset."""
    path = _aligned_checkpoint(tmp_dataset, tmp_path)
    out = tmp_path / "m.json"

    eval_script.run_eval(["--checkpoint", str(path), "--rendering", "anycam4d", "--output", str(out)])

    assert "cc_psnr" not in json.loads(out.read_text())["mean"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
