"""Provenance manifest builder for ``PrepWriter`` outputs."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np

from anycam4d.provenance import git_sha


if TYPE_CHECKING:
    from collections.abc import Callable


def build_manifest(
    *,
    source: str,
    importer: str,
    cli_args: dict,
    variant: str | None = None,
    stages: list[str] | None = None,
    extra_tools: dict | None = None,
    source_url: str | None = None,
    source_hash: str | None = None,
) -> dict:
    """Build a best-effort provenance manifest for a ``PrepWriter`` output directory.

    Never raises: every probe (git SHA, tool versions) is wrapped so an environment quirk
    degrades a field to ``"unknown"`` rather than failing the whole export. Written as the
    ``prep_manifest.json`` sidecar by ``PrepWriter.finalize`` — never into the dataset JSON
    itself, since that would require a schema change.

    Args:
        source: Human-readable description of the raw data this dataset was built from.
        importer: Name of the importer or variant-builder that produced this dataset.
        cli_args: The CLI arguments the run was invoked with, for exact reproduction.
        variant: Variant name, when this is a post-hoc variant rather than a fresh import.
        stages: Names of the ``Stage`` pipeline applied, in order.
        extra_tools: Additional tool versions to merge in (e.g. a caller-supplied pycolmap rev).
        source_url: Canonical URL the raw source was fetched from (registry-driven runs).
        source_hash: ``sha256:...`` fingerprint of the fetched source, for exact reproduction.

    Returns:
        A JSON-serializable manifest dict.
    """
    tools = {
        "anycam4d_git_sha": git_sha(),
        "opencv": _safe_version(lambda: cv2.__version__),
        "numpy": _safe_version(lambda: np.__version__),
    }
    if extra_tools:
        tools.update(extra_tools)

    return {
        "source": source,
        "source_url": source_url,
        "source_hash": source_hash,
        "importer": importer,
        "cli_args": cli_args,
        "variant": variant,
        "stages": stages,
        "created_utc": datetime.now(UTC).isoformat(),
        "tools": tools,
    }


def _safe_version(getter: Callable[[], str]) -> str:
    """Return ``getter()``, or ``"unknown"`` if it raises."""
    try:
        return str(getter())
    except Exception:  # noqa: BLE001
        return "unknown"
