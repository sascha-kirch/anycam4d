"""Staging tests: markers and input hashing (the in-place resume mechanism)."""

from __future__ import annotations

from pathlib import Path

from anycam4d.data.prep.staging import input_hash
from anycam4d.data.prep.staging import mark_done
from anycam4d.data.prep.staging import stage_done


def test_changed_input_hash_invalidates(tmp_path: Path) -> None:
    state = tmp_path / "prep_state.json"
    mark_done(state, "canonical", input_hash("src", 1), {})
    assert not stage_done(state, "canonical", input_hash("src", 2))


def test_stage_done_false_when_artifact_missing(tmp_path: Path) -> None:
    state = tmp_path / "prep_state.json"
    h = input_hash("x")
    mark_done(state, "canonical", h, {})
    missing = tmp_path / "artifact"
    assert not stage_done(state, "canonical", h, artifact=missing)
    missing.mkdir()
    assert stage_done(state, "canonical", h, artifact=missing)


def test_input_hash_stable_and_sensitive() -> None:
    a = input_hash("src", 1, {"k": "v"})
    assert a == input_hash("src", 1, {"k": "v"})
    assert a != input_hash("src", 2, {"k": "v"})
    assert a != input_hash("src", 1, {"k": "w"})


def test_input_hash_path_sensitive_to_content(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a")
    h1 = input_hash(f)
    f.write_text("much longer content changes size")
    assert input_hash(f) != h1


def test_unmarked_artifact_is_not_done(tmp_path: Path) -> None:
    """A half-built (never-marked) artifact resumes rather than reading as complete."""
    state = tmp_path / "prep_state.json"
    h = input_hash("x")
    target = tmp_path / "art"
    target.mkdir()  # partial output present, but mark_done never ran
    assert not stage_done(state, "base", h, artifact=target)
    mark_done(state, "base", h, {})
    assert stage_done(state, "base", h, artifact=target)
