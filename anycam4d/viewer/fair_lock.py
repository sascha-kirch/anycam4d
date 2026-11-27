"""First-come, first-served lock shared by the training loop and the viewer's render threads."""

from __future__ import annotations

import threading
from typing import Self


class FairLock:
    """A lock that passes to the longest waiter on release.

    ``threading.Lock`` lets the releasing thread take it straight back, so a training loop that re-acquires it every
    step starves the render thread and the view freezes.
    """

    def __init__(self) -> None:
        """Start with no holder and no waiters."""
        self._cond = threading.Condition(threading.Lock())
        self._next_ticket = 0
        self._serving = 0

    def acquire(self) -> bool:
        """Block until every earlier caller has released."""
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving:
                self._cond.wait()
        return True

    def release(self) -> None:
        """Hand the lock to the next caller in line."""
        with self._cond:
            self._serving += 1
            self._cond.notify_all()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
