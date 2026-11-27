"""Shared console and progress UI: one rich ``Console`` so concurrent bars never fight over the terminal.

``progress_bar`` (per finished unit), ``spinner`` (opaque long work such as COLMAP), ``download_bar`` (bytes, speed,
ETA; a ``urlretrieve`` reporthook) and ``notice`` (one-off lines). Depends on rich only.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


console = Console()


@contextmanager
def progress_bar(description: str, total: int, *, unit: str = "") -> Iterator[Callable[..., None]]:
    """Yield an ``advance(n=1)`` callable driving a determinate bar with spinner + elapsed clock.

    The spinner animates continuously (rich refreshes on its own thread), so the display shows
    the task is alive even between completions of long-running units of work. ``unit`` labels the
    count (e.g. ``"frames"`` renders ``3/10 frames``).
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(unit),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=total)

        def advance(n: int = 1) -> None:
            progress.advance(task, n)

        yield advance


@contextmanager
def spinner(message: str) -> Iterator[None]:
    """Show an animated spinner + message for an opaque long-running step (e.g. COLMAP)."""
    with console.status(message, spinner="dots"):
        yield


@contextmanager
def download_bar() -> Iterator[Callable[[int, int, int], None]]:
    """Yield a ``urlretrieve`` ``reporthook`` driving a byte-counting bar (size + speed + ETA)."""
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("download", total=None)

        def hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress.update(task, total=total_size)
            progress.update(task, completed=block_num * block_size)

        yield hook


def notice(message: str) -> None:
    """Print a one-off informational notice on the shared console."""
    console.print(message)


def warn(message: str) -> None:
    """Print a warning on the shared console (yellow), prefixed with a warning glyph."""
    console.print(f"[yellow]⚠ {message}[/]")


def rule(title: str, *, style: str = "cyan") -> None:
    """Draw a full-width horizontal rule with a centered ``title`` (section separator)."""
    console.rule(f"[{style}]{title}[/]", style=style)
