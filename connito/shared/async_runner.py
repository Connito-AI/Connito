"""Persistent asyncio event loop on a daemon thread.

Used by sync code paths in the validator that now hold an
`AsyncSubtensor` (the lite endpoint). Each chain RPC is a coroutine; the
runner schedules it on its loop and the calling thread blocks for the
result, mirroring the previous sync behaviour without making the whole
main loop async.

Single instance per process (one persistent connection cache, one
in-flight queue). The `ChainSubmitter` owns this runner so its async
`set_weights` and `set_commitment` calls share one loop and don't race.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Awaitable, TypeVar

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class AsyncRunner:
    """Drive coroutines on a dedicated daemon thread.

    Construction blocks until the loop is up; `stop()` joins the thread.
    `run(coro)` blocks the caller until the coroutine resolves;
    `submit(coro)` is fire-and-forget.
    """

    def __init__(self, name: str = "connito-async-runner") -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._stop_called = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        if not self._loop_ready.wait(timeout=10):
            raise RuntimeError(f"{name}: event loop failed to start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                # Cancel any leftover tasks so close() doesn't warn.
                tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in tasks:
                    t.cancel()
                if tasks:
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("AsyncRunner: loop not started")
        return self._loop

    def run(self, coro: Awaitable[T]) -> T:
        """Schedule `coro` and block the caller until it resolves."""
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("AsyncRunner: loop is not running")
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def submit(self, coro: Awaitable[T]) -> Future:
        """Schedule `coro` without blocking; returns a Future the caller can poll."""
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("AsyncRunner: loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        if self._stop_called:
            return
        self._stop_called = True
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=10)
