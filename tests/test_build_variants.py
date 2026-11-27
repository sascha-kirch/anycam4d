"""``build_variants`` orchestration: order, resume, force and scope flags, dry run, download seam (imports stubbed)."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest


# scripts/ is not an importable package — load the orchestrator by file path (like test_eval_script).
_BV_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_datasets" / "build_variants.py"
_spec = importlib.util.spec_from_file_location("build_variants_script", _BV_PATH)
assert _spec is not None and _spec.loader is not None
bv = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bv
_spec.loader.exec_module(bv)


def _write_registry(path: Path, *, variants: list[str] | None = None, scales: list[float] | None = None) -> None:
    variants = variants if variants is not None else ["native", "undistorted"]
    scales = scales if scales is not None else [1, 2]
    var_yaml = ", ".join(variants)
    scale_yaml = ", ".join(str(s) for s in scales)
    path.write_text(
        "dataset: dynerf\n"
        "license: { redistribute: true, source: test }\n"
        "source_url: https://example.com/src\n"
        "scenes:\n"
        "  s0:\n"
        "    source_subdir: scene0\n"
        "    fps: 30\n"
        "    eval_cams: [0]\n"
        f"    variants: [{var_yaml}]\n"
        f"    scales: [{scale_yaml}]\n"
    )


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fixture registry + fake source + stubbed importers; returns (make_args, calls)."""
    reg = tmp_path / "reg.yaml"
    _write_registry(reg)
    source_root = tmp_path / "src"
    (source_root / "scene0").mkdir(parents=True)
    (source_root / "scene0" / "raw.bin").write_text("data")
    out_root = tmp_path / "out"

    calls: list[tuple] = []
    builds = itertools.count()

    def stub_base(importer, out_dir, *, eval_cams=(), manifest=None, **kw) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        dataset_json = Path(out_dir) / "train_dataset.json"
        dataset_json.write_text("{}")
        # A real build takes seconds; on a fast machine two stub builds share one mtime tick.
        built_at = time.time_ns() + next(builds) * 10**9
        os.utime(dataset_json, ns=(built_at, built_at))
        if manifest is not None:
            (Path(out_dir) / "prep_manifest.json").write_text(json.dumps(manifest))
        calls.append(("base", manifest["source"] if manifest else None))

    def stub_variant(base_dir, out_dir, spec, jobs=1) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "train_dataset.json").write_text("{}")
        (Path(out_dir) / "prep_manifest.json").write_text(json.dumps({"source": str(base_dir), "variant": spec.name}))
        calls.append(("variant", Path(out_dir).name))

    monkeypatch.setattr(bv, "build_dataset", stub_base)
    monkeypatch.setattr(bv, "build_variant", stub_variant)

    def make_args(**over) -> argparse.Namespace:
        base = {
            "registry": reg,
            "scene": None,
            "source_root": source_root,
            "out_root": out_root,
            "work_dir": None,
            "jobs": 1,
            "force": False,
            "from_stage": None,
            "only": None,
            "dry_run": False,
            "colmap_threads": 4,
            "colmap_max_image_size": 3200,
            "colmap_gpu": True,
            "dense": True,
            "dense_max_image_size": 1600,
        }
        base.update(over)
        return argparse.Namespace(**base)

    return make_args, calls, out_root, reg


def _clear_marker(state_path: Path, stage: str) -> None:
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.pop(stage, None)
        state_path.write_text(json.dumps(state))


def _base_calls(calls: list[tuple]) -> list[tuple]:
    return [c for c in calls if c[0] == "base"]


def _variant_names(calls: list[tuple]) -> list[str]:
    return [c[1] for c in calls if c[0] == "variant"]


# variants [native, undistorted] x scales [1, 2], minus the redundant native-at-full-res (== base).
_ALL_VARIANT_DIRS = ["s0_r2", "s0_undistorted", "s0_undistorted_r2"]


def test_full_build(env) -> None:
    make_args, calls, out_root, _ = env
    bv.run(make_args())

    assert (out_root / "s0" / "train_dataset.json").exists()
    for name in _ALL_VARIANT_DIRS:
        assert (out_root / name / "train_dataset.json").exists()
    # native at full res is NOT a separate dir (the base dataset is native)
    assert not (out_root / "s0_native").exists()

    assert len(_base_calls(calls)) == 1
    assert sorted(_variant_names(calls)) == _ALL_VARIANT_DIRS

    manifest = json.loads((out_root / "s0" / "prep_manifest.json").read_text())
    assert manifest["source_url"] == "https://example.com/src"
    assert manifest["source_hash"].startswith("sha256:")


def test_pinhole_base_skips_undistorted(env, monkeypatch) -> None:
    make_args, calls, out_root, reg = env
    _write_registry(reg, variants=["native", "undistorted"], scales=[2])

    def stub_base_pinhole(importer, out_dir, *, eval_cams=(), manifest=None, **kw) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "train_dataset.json").write_text(json.dumps({"cameras": [{"model": "pinhole"}]}))
        calls.append(("base", None))

    monkeypatch.setattr(bv, "build_dataset", stub_base_pinhole)
    bv.run(make_args())
    # undistorting a pinhole base would duplicate native, so only the native downscale is derived.
    assert _variant_names(calls) == ["s0_r2"]


def test_resume_is_noop(env) -> None:
    make_args, calls, _, _ = env
    bv.run(make_args())
    calls.clear()
    bv.run(make_args())
    assert calls == []  # every stage skipped


def test_delete_base_marker_rebuilds_base_and_variants(env) -> None:
    make_args, calls, out_root, _ = env
    bv.run(make_args())
    _clear_marker(out_root / "s0.prep_state.json", "base")
    calls.clear()
    bv.run(make_args())
    assert len(_base_calls(calls)) == 1
    assert sorted(_variant_names(calls)) == _ALL_VARIANT_DIRS  # rebuilt because base changed


def test_per_variant_isolation(env) -> None:
    make_args, calls, out_root, _ = env
    bv.run(make_args())
    # drop one variant artifact; its siblings and the base dataset stay valid
    shutil.rmtree(out_root / "s0_undistorted_r2")
    calls.clear()
    bv.run(make_args())
    assert _base_calls(calls) == []
    assert _variant_names(calls) == ["s0_undistorted_r2"]


def test_from_base_redoes_base_onward(env) -> None:
    make_args, calls, _, _ = env
    bv.run(make_args())
    calls.clear()
    bv.run(make_args(from_stage="base"))
    assert len(_base_calls(calls)) == 1
    assert sorted(_variant_names(calls)) == _ALL_VARIANT_DIRS


def test_from_variants_leaves_base(env) -> None:
    make_args, calls, _, _ = env
    bv.run(make_args())
    calls.clear()
    bv.run(make_args(from_stage="variants"))
    assert _base_calls(calls) == []  # base is an earlier stage, still skip-if-done
    assert sorted(_variant_names(calls)) == _ALL_VARIANT_DIRS


def test_force_redoes_all(env) -> None:
    make_args, calls, _, _ = env
    bv.run(make_args())
    calls.clear()
    bv.run(make_args(force=True))
    assert len(_base_calls(calls)) == 1
    assert sorted(_variant_names(calls)) == _ALL_VARIANT_DIRS


def test_only_variants_skips_base(env) -> None:
    make_args, calls, out_root, _ = env
    bv.run(make_args())
    _clear_marker(out_root / "s0.prep_state.json", "base")
    calls.clear()
    bv.run(make_args(only="variants"))
    assert _base_calls(calls) == []  # base excluded by --only even though its marker was cleared
    assert _variant_names(calls) == []  # variants already done -> skip-if-done


def test_only_with_force_scopes_to_one_variant(env) -> None:
    make_args, calls, _, _ = env
    bv.run(make_args())
    calls.clear()
    bv.run(make_args(only="native", force=True))  # bare variant name selects every scale tier of it
    assert _base_calls(calls) == []
    assert _variant_names(calls) == ["s0_r2"]  # native at full res is the base, so only the r2 tier


def test_only_base_with_force_redoes_just_base(env) -> None:
    make_args, calls, _, _ = env
    bv.run(make_args())
    calls.clear()
    bv.run(make_args(only="base", force=True))
    assert len(_base_calls(calls)) == 1
    assert _variant_names(calls) == []


def test_dry_run_writes_nothing(env) -> None:
    make_args, calls, out_root, _ = env
    bv.run(make_args(dry_run=True))
    assert calls == []
    assert not out_root.exists() or not (out_root / "s0").exists()


def test_download_fetcher_seam_and_hash(env) -> None:
    make_args, _, _, reg = env
    cfg = bv.load_registry(reg)
    root = reg.parent / "src"  # the fixture source root
    dest = reg.parent / "fetched"
    fetcher = bv.download.LocalDirFetcher(root)
    path, digest = bv.download.fetch_source(cfg, "s0", dest, fetcher=fetcher)
    assert path.exists()
    assert digest.startswith("sha256:")
    assert digest == bv.download.fetch_source(cfg, "s0", dest, fetcher=fetcher)[1]


def test_only_a_variant_does_not_fetch_the_raw_source(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--only <variant>`` never fetches the raw corpus; only the base stage reads it."""
    make_args, calls, out_root, _ = env
    bv.run(make_args())  # build everything once, from the real fixture source
    calls.clear()

    def explode(*a, **kw):
        raise AssertionError("fetch_source must not be called for a variant-only run")

    monkeypatch.setattr(bv.download, "fetch_source", explode)
    missing_source = out_root.parent / "source-is-gone"

    bv.run(make_args(source_root=missing_source, only="undistorted", force=True))

    assert _base_calls(calls) == []
    assert _variant_names(calls) == ["s0_undistorted", "s0_undistorted_r2"]


def test_out_of_scope_base_is_not_reported_as_already_done(env, capsys) -> None:
    """A stage excluded by ``--only`` reads as out of scope, not as already done."""
    make_args, _, _, _ = env
    _clear_marker((make_args().out_root / "s0.prep_state.json"), "base")

    bv.run(make_args(only="undistorted", force=True))

    out = capsys.readouterr().out
    assert "out of scope" in out
    assert "already done" not in out.split("out of scope")[0]
