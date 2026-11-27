"""Code provenance: the recorded commit is real, degrades instead of raising, and reaches TensorBoard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from anycam4d import provenance
from anycam4d.provenance import code_provenance
from anycam4d.provenance import git_branch
from anycam4d.provenance import git_dirty
from anycam4d.provenance import git_sha
from anycam4d.provenance import git_untracked

from .test_checkpoint_config import _make_trainer


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Trainer.setup builds a CUDA backend")

SHA_LENGTH = 40


def test_git_sha_is_a_real_commit() -> None:
    """The package tree is a git checkout, so the probe returns a full SHA."""
    sha = git_sha()
    assert len(sha) == SHA_LENGTH
    assert all(c in "0123456789abcdef" for c in sha)


def test_dirty_ignores_untracked_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An untracked file makes the tree untracked, never dirty."""
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> str:
        calls.append(args)
        # What `git status --porcelain` prints for a tree whose only anomaly is an untracked file.
        return "?? scratch.txt" if "-uno" not in args else ""

    monkeypatch.setattr(provenance, "_run_git", fake_git)
    assert git_dirty() is False
    assert git_untracked() is True
    # Scoped by pathspec to the project tree: `git status` is repo-wide regardless of cwd.
    dirty_call = next(c for c in calls if "-uno" in c)
    assert dirty_call[:3] == ("status", "--porcelain", "-uno")
    assert dirty_call[3] == "--"
    assert dirty_call[4].endswith("/anycam4d"), dirty_call[4]
    assert not dirty_call[4].endswith("/anycam4d/anycam4d"), "pathspec must cover conf/ and scripts/"


def test_probe_failure_degrades_rather_than_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without git every field degrades; ``git_dirty`` is ``None``, not clean."""
    monkeypatch.setattr(provenance, "_run_git", lambda *_: None)
    assert git_sha() == "unknown"
    assert git_branch() == "unknown"
    assert git_dirty() is None
    assert git_untracked() is None
    assert code_provenance() == {
        "git_sha": "unknown",
        "git_branch": "unknown",
        "git_dirty": None,
        "git_untracked": None,
    }


@requires_cuda
def test_run_records_provenance_in_json_tb_and_checkpoint(tmp_dataset: Path, tmp_path: Path) -> None:
    """A run records its commit in JSON, TensorBoard and the checkpoint."""
    result_dir = tmp_path / "t"
    trainer = _make_trainer(tmp_dataset, result_dir, config_snapshot=None)
    ckpt_path = trainer.save_checkpoint(0)
    trainer.writer.close()

    expected = code_provenance()

    written = json.loads((result_dir / "run_provenance.json").read_text())
    assert written == expected

    ckpt = torch.load(ckpt_path, weights_only=False)
    assert ckpt["provenance"] == expected

    accumulator = EventAccumulator(str(result_dir / "tb"))
    accumulator.Reload()
    logged = accumulator.Tensors("provenance/code/text_summary")[0].tensor_proto.string_val[0].decode()
    assert expected["git_sha"] in logged
    assert expected["git_branch"] in logged
