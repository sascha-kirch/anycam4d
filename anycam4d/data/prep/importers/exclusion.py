"""Source-level camera exclusion, shared by every video importer.

Some rigs carry a view that hurts every downstream stage — out of focus, mis-synced, smeared lens.
Dropping it *at discovery* (rather than filtering it out of the finished dataset) means the camera
never reaches calibration: no COLMAP feature matching, no dense MVS depth map, no ffmpeg
extraction, no entry in the emitted manifests. The reconstruction is then solved from good views
only, so the bad camera cannot pull the sparse cloud or the recovered poses with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anycam4d.console import notice


if TYPE_CHECKING:
    from collections.abc import Iterable


def drop_excluded(cam_ids: Iterable[int], exclude: Iterable[int], *, source: str) -> list[int]:
    """Return ``cam_ids`` without the excluded ids, reporting what was dropped.

    Args:
        cam_ids: Camera ids discovered in the source.
        exclude: Camera ids to remove.
        source: Human-readable source description, used in the error when nothing survives.

    Returns:
        The sorted surviving camera ids.

    Raises:
        ValueError: If the exclusion list would remove every camera.
    """
    excluded = set(exclude)
    kept = sorted(cid for cid in cam_ids if cid not in excluded)

    dropped = sorted(set(cam_ids) & excluded)
    if dropped:
        notice(f"[yellow]Excluding camera(s) {dropped} — not calibrated, extracted, or written.[/]")
    missing = sorted(excluded - set(cam_ids))
    if missing:
        notice(f"[yellow]excluded_cams lists camera(s) {missing} that are not present in the source.[/]")
    if not kept:
        raise ValueError(f"every camera in {source} is excluded ({sorted(excluded)}); nothing left to import")
    return kept
