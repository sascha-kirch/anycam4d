r"""Fetch a dataset's raw source **to the user's own machine** — never re-hosts.

Each dataset is fetched from its original home (DyNeRF: GitHub release; SelfCap: Hugging Face; DeepView:
per-scene URLs), respecting its license, which is printed before any bytes move. Third-party data is never
re-hosted; Dynopsis is ours and downloads already prepared from Hugging Face.

The fetch is behind a small ``FetcherProto`` seam so unit tests inject a local-directory fetcher
(no network); the real network fetchers are ``@pytest.mark.slow`` / run by hand. Every fetch returns
a ``sha256:`` ``source_hash`` (over a size+path manifest of the fetched tree) for provenance.

Usage::

    pixi run download --registry conf/datasets/dynerf.yaml --scene flame_steak --dest ~/datasets/DyNeRF
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

from anycam4d.console import download_bar
from anycam4d.console import notice
from anycam4d.data.prep.registry import check_scene
from anycam4d.data.prep.registry import load_registry


if TYPE_CHECKING:
    from anycam4d.data.prep.registry import DatasetConfig


@runtime_checkable
class FetcherProto(Protocol):
    """Fetch one scene's raw source into ``dest`` and return the fetched scene directory."""

    def __call__(self, cfg: DatasetConfig, scene: str, dest: Path) -> Path:
        """Download ``scene`` of ``cfg`` under ``dest``; return the scene's source root."""
        ...


def source_hash(path: Path) -> str:
    """``sha256:`` over a size+path manifest of a file or directory tree (no multi-GB reads)."""
    hasher = hashlib.sha256()
    if path.is_file():
        st = path.stat()
        hasher.update(f"{path.name}:{st.st_size}".encode())
    else:
        for child in sorted(path.rglob("*")):
            if child.is_file():
                hasher.update(f"{child.relative_to(path)}:{child.stat().st_size}\0".encode())
    return f"sha256:{hasher.hexdigest()}"


def _download(url: str, archive: Path) -> None:
    """Stream ``url`` to ``archive`` with a live progress bar (size + speed + ETA)."""
    notice(f"  downloading {url}")
    with download_bar() as hook:
        urllib.request.urlretrieve(url, archive, reporthook=hook)  # noqa: S310 — https asset
    notice(f"  downloaded {archive.stat().st_size / 1e6:.1f} MB")


def _extract(archive: Path, dest: Path, subdir: str) -> Path:
    """Unpack ``archive`` so its content lands at ``dest/subdir`` exactly once, then delete it.

    Handles both archive layouts: one that already wraps everything in a top-level ``<subdir>/``
    folder (kept as-is, no double nesting) and a flat one (wrapped into ``<subdir>/``). Reports the
    file count and removes the archive on success.
    """
    notice(f"  extracting {archive.name} ...")
    staging = dest / f".extract_{subdir}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.unpack_archive(archive, staging)

    target = dest / subdir
    if target.exists():
        shutil.rmtree(target)
    entries = list(staging.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        entries[0].rename(target)  # archive already had a single top-level folder -> avoid nesting
        shutil.rmtree(staging)
    else:
        staging.rename(target)  # flat archive -> the staging dir becomes <subdir>/

    n_files = sum(1 for p in target.rglob("*") if p.is_file())
    archive.unlink()
    notice(f"  extracted {n_files} files -> {target}; removed archive {archive.name}")
    return target


class LocalDirFetcher(FetcherProto):
    """Test/offline fetcher: copy an already-present source tree from ``root`` into ``dest``."""

    def __init__(self, root: Path) -> None:
        """Store the local source root to copy scenes from."""
        self._root = Path(root)

    def __call__(self, cfg: DatasetConfig, scene: str, dest: Path) -> Path:
        """Copy the scene's source subdir from ``root`` into ``dest`` and return it."""
        sc = cfg.scenes[scene]
        src = self._root / sc.source_subdir
        target = dest / sc.source_subdir
        if src.resolve() != target.resolve():
            shutil.copytree(src, target, dirs_exist_ok=True)
        return target


def _fetch_zip(cfg: DatasetConfig, scene: str, dest: Path) -> Path:
    """Download ``source_url/<subdir>.zip`` and extract it under ``dest`` (DyNeRF / DeepView)."""
    sc = cfg.scenes[scene]
    url = f"{cfg.source_url.rstrip('/')}/{sc.source_subdir}.zip"
    archive = dest / f"{sc.source_subdir}.zip"
    dest.mkdir(parents=True, exist_ok=True)
    _download(url, archive)
    if sc.split_parts:
        archive = _join_split_zip(cfg, sc.source_subdir, archive, sc.split_parts)
    return _extract(archive, dest, sc.source_subdir)


def _join_split_zip(cfg: DatasetConfig, subdir: str, archive: Path, n_parts: int) -> Path:
    """Fetch the ``.z01..zNN`` parts next to ``archive`` and join them into one zip (``zip -s 0``)."""
    if shutil.which("zip") is None:
        raise RuntimeError(f"{subdir} is a split zip archive; joining it needs the `zip` binary on PATH")
    parts = [archive.with_suffix(f".z{i:02d}") for i in range(1, n_parts + 1)]
    for part in parts:
        _download(f"{cfg.source_url.rstrip('/')}/{part.name}", part)
    if archive.with_suffix(f".z{n_parts + 1:02d}").exists():
        raise RuntimeError(f"{subdir}: more than {n_parts} split parts on disk; fix split_parts in the registry")
    joined = archive.with_name(f"{subdir}_joined.zip")
    # zip waits forever for a missing part instead of failing; the timeout turns that into an error.
    cmd = ["zip", "-q", "-s", "0", str(archive), "--out", str(joined)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=3600)  # noqa: S603
    for path in [*parts, archive]:
        path.unlink()
    return joined


def _fetch_dynerf(cfg: DatasetConfig, scene: str, dest: Path) -> Path:
    """Fetch a DyNeRF scene archive from the GitHub release and extract it under ``dest``."""
    return _fetch_zip(cfg, scene, dest)


def _hf_pull_archive(repo_id: str, filename: str, dest: Path, subdir: str) -> Path:
    """Download one HF ``.tar.gz`` (with the hub's own progress bar) and extract it under ``dest``."""
    notice(f"  downloading {repo_id}::{filename}")
    try:
        archive = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=dest))
    except RepositoryNotFoundError as e:  # also GatedRepoError: not logged in, or terms not accepted
        raise RuntimeError(
            f"no access to the gated dataset {repo_id}: run `hf auth login`, then accept the terms at "
            f"https://huggingface.co/datasets/{repo_id} (Files and versions tab) and retry"
        ) from e
    return _extract(archive, dest, subdir)


def _fetch_selfcap(cfg: DatasetConfig, scene: str, dest: Path) -> Path:
    """Fetch a SelfCap scene (and its separate calib archive, if any) from HuggingFace into ``dest``.

    SelfCap ships each release as a root-level ``<subdir>.tar.gz`` (e.g. ``yoga-release.tar.gz`` +
    ``yoga-calib.tar.gz``), not as a directory tree — so we pull the specific archive(s) and unpack.
    """
    sc = cfg.scenes[scene]
    dest.mkdir(parents=True, exist_ok=True)
    target = _hf_pull_archive(cfg.source_url, f"{sc.source_subdir}.tar.gz", dest, sc.source_subdir)
    if sc.calib_dir is not None:
        _hf_pull_archive(cfg.source_url, f"{sc.calib_dir}.tar.gz", dest, sc.calib_dir)
    return target


def _fetch_deepview(cfg: DatasetConfig, scene: str, dest: Path) -> Path:
    """Fetch a DeepView scene archive from its per-scene URL (``source_url/<subdir>.zip``)."""
    return _fetch_zip(cfg, scene, dest)


def _fetch_dynopsis(cfg: DatasetConfig, scene: str, dest: Path) -> Path:
    """Fetch one prepared Dynopsis set (``<source_subdir>.tar`` on Hugging Face), unpacked to ``dest/<source_subdir>``."""
    subdir = cfg.scenes[scene].source_subdir
    notice(f"  downloading {cfg.source_url}::{subdir}.tar")
    archive = Path(hf_hub_download(cfg.source_url, f"{subdir}.tar", repo_type="dataset", local_dir=dest))
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()
    return dest / subdir


DEFAULT_FETCHERS: dict[str, FetcherProto] = {
    "dynerf": _fetch_dynerf,
    "selfcap": _fetch_selfcap,
    "deepview": _fetch_deepview,
    "dynopsis": _fetch_dynopsis,
}


def fetch_source(
    cfg: DatasetConfig,
    scene: str,
    dest: Path,
    fetcher: FetcherProto | None = None,
) -> tuple[Path, str]:
    """Print the license/source, fetch ``scene`` into ``dest``, and return ``(path, source_hash)``."""
    check_scene(cfg, scene)
    notice(
        f"[bold]{cfg.dataset}/{scene}[/] — source: {cfg.source_url}\n"
        f"  license: {cfg.license.source} (redistribute={cfg.license.redistribute})\n"
        f"  fetching to your machine; respect the dataset license."
    )
    fetch = fetcher if fetcher is not None else DEFAULT_FETCHERS[cfg.dataset]
    path = fetch(cfg, scene, dest)
    return path, source_hash(path)


def main() -> None:
    """Parse CLI args and fetch one (or all) scene(s) of a registry to a user path."""
    parser = argparse.ArgumentParser(description="Fetch a dataset's raw source to your own machine.")
    parser.add_argument("--registry", type=Path, required=True, help="Dataset registry YAML (conf/datasets/*.yaml).")
    parser.add_argument("--scene", type=str, default=None, help="Scene to fetch (default: every scene).")
    parser.add_argument("--dest", type=Path, required=True, help="Destination root for the raw source.")
    args = parser.parse_args()

    cfg = load_registry(args.registry)
    scenes = [args.scene] if args.scene is not None else list(cfg.scenes)
    for scene in scenes:
        path, digest = fetch_source(cfg, scene, args.dest)
        notice(f"  fetched {scene} -> {path}  ({digest})")


if __name__ == "__main__":
    main()
