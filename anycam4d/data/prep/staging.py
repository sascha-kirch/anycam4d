"""Resume markers for ``build_variants``: skip-if-done per stage, keyed by an input hash.

``prep_state.json`` records ``{done, input_hash, tool_versions, timestamp}`` per stage. A stage is skipped iff it is
marked done, its input hash still matches and its output directory exists. Stages build in place and write the
marker only after success, so a crash leaves an unmarked stage that the next run redoes. Inputs are fingerprinted by
``(relpath, size, mtime)``, not content. Torch-free.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC
from datetime import datetime
from pathlib import Path


def input_hash(*parts: object) -> str:
    """SHA-256 over a stage's inputs. ``Path`` args are fingerprinted by size+mtime, not content."""
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, Path):
            hasher.update(_path_fingerprint(part).encode())
        else:
            hasher.update(repr(part).encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _path_fingerprint(path: Path) -> str:
    """Cheap fingerprint of a file or directory: ``(relpath, size, mtime_ns)`` of every file."""
    if not path.exists():
        return f"missing:{path}"
    if path.is_file():
        st = path.stat()
        return f"file:{path.name}:{st.st_size}:{st.st_mtime_ns}"
    entries = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            st = child.stat()
            entries.append(f"{child.relative_to(path)}:{st.st_size}:{st.st_mtime_ns}")
    return "dir:" + "|".join(entries)


def _load_state(state_path: Path) -> dict:
    """Read ``prep_state.json``, or an empty dict if it is absent or unreadable."""
    if not state_path.exists():
        return {}
    try:
        with state_path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def stage_done(state_path: Path, stage: str, expected_hash: str, artifact: Path | None = None) -> bool:
    """True iff ``stage`` is marked done with a matching input hash (and its artifact still exists)."""
    entry = _load_state(state_path).get(stage)
    if entry is None or not entry.get("done") or entry.get("input_hash") != expected_hash:
        return False
    return artifact is None or artifact.exists()


def mark_done(state_path: Path, stage: str, stage_hash: str, tool_versions: dict) -> None:
    """Record ``stage`` as complete with its input hash and tool versions (atomic JSON rewrite)."""
    state = _load_state(state_path)
    state[stage] = {
        "done": True,
        "input_hash": stage_hash,
        "tool_versions": tool_versions,
        "timestamp": _now_iso(),
    }
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


def _now_iso() -> str:
    """UTC timestamp for the marker."""
    return datetime.now(UTC).isoformat()
