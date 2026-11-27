"""Code provenance for a run: commit, branch, and whether tracked or untracked files differ.

A modified tracked file means the SHA does not describe the code that ran; an untracked file does not.
Every probe degrades to ``"unknown"`` or ``None`` instead of failing training.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent
# The project tree, not the import package: `conf/`, `scripts/` and `tests/` sit beside `anycam4d/`
# and a modified group preset changes the composition as surely as a modified module changes the
# code. Used as the pathspec that scopes the dirty probes; see git_dirty.
_PROJECT_ROOT = _PACKAGE_ROOT.parent


def _run_git(*args: str) -> str | None:
    """Run a git command inside the package tree, or return None if git cannot answer."""
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell, arguments are literals from this module
            ["git", *args],  # noqa: S607 — resolved from PATH deliberately; this is a best-effort probe
            capture_output=True,
            text=True,
            check=True,
            cwd=_PACKAGE_ROOT,
        )
    except Exception:  # noqa: BLE001 — any failure means "no provenance available", never a crash
        return None
    return result.stdout.strip()


def git_sha() -> str:
    """Return the full commit SHA of the working tree, or ``"unknown"``."""
    return _run_git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool | None:
    """Whether tracked files in the project tree are modified, or ``None`` if undeterminable.

    Scoped by pathspec to the project directory, so an enclosing repository's other files do not count.
    ``None`` must not read as clean.
    """
    status = _run_git("status", "--porcelain", "-uno", "--", str(_PROJECT_ROOT))
    return None if status is None else bool(status)


def git_untracked() -> bool | None:
    """Whether the project tree holds untracked, non-ignored files, or ``None``; informational only."""
    status = _run_git("status", "--porcelain", "--untracked-files=normal", "--", str(_PROJECT_ROOT))
    if status is None:
        return None
    return any(line.startswith("??") for line in status.splitlines())


def git_branch() -> str:
    """Return the current branch name, ``"detached"`` on a detached HEAD, or ``"unknown"``."""
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    if branch is None:
        return "unknown"
    return "detached" if branch == "HEAD" else branch


def code_provenance() -> dict[str, str | bool | None]:
    """Return the commit, branch, tracked-dirty flag and untracked-files flag for this run."""
    return {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "git_dirty": git_dirty(),
        "git_untracked": git_untracked(),
    }
