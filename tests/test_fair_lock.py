"""The viewer lock hands over to a waiting render thread instead of letting the training loop re-take it."""

from __future__ import annotations

import threading
import time

from anycam4d.viewer.fair_lock import FairLock


def test_waiting_thread_gets_the_lock_while_another_loops_on_it() -> None:
    """A loop that re-acquires every 5 ms must not starve a second thread (the frozen-viewer bug)."""
    lock = FairLock()
    stop = threading.Event()

    def trainer() -> None:
        while not stop.is_set():
            with lock:
                time.sleep(0.005)

    thread = threading.Thread(target=trainer, daemon=True)
    thread.start()
    try:
        waits = []
        for _ in range(10):
            start = time.monotonic()
            with lock:
                waits.append(time.monotonic() - start)
        assert max(waits) < 0.5
    finally:
        stop.set()
        thread.join(timeout=1.0)
