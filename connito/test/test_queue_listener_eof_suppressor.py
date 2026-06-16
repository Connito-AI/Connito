"""Pin the contract of `_install_queue_listener_eof_suppressor`.

The bug it silences leaks from a stdlib QueueListener thread inside
the HF download subprocess. We don't reproduce the full HF chain in
unit tests — that needs network and a HuggingFace subprocess. Instead
we directly invoke the installed `threading.excepthook` with the same
shape of args the real failure produces and verify:

  1. EOFError with a thread named `Thread-N (_monitor)` is swallowed
     (the original fallback hook is NOT invoked).
  2. EOFError in any other thread name surfaces to the fallback.
  3. RuntimeError (or any non-EOFError) in a `_monitor` thread
     surfaces to the fallback — we don't blanket-suppress monitor
     threads.

`threading.excepthook` looks up `threading.__excepthook__` at install
time and stores it as the fallback. To capture what would have
escaped, we patch `threading.__excepthook__` to a recording function
BEFORE installing the suppressor, then synthesise `ExceptHookArgs`
and invoke the installed hook directly.
"""
from __future__ import annotations

import threading
import types

import pytest

from connito.shared.hf_distribute import _install_queue_listener_eof_suppressor


@pytest.fixture
def installed_hook(monkeypatch):
    """Install the suppressor on top of a recording fallback; return
    (hook, recorded) so the test can synthesise an args tuple and
    call the hook directly.
    """
    recorded: list[threading.ExceptHookArgs] = []

    def _fallback(args):
        recorded.append(args)

    monkeypatch.setattr(threading, "__excepthook__", _fallback)
    monkeypatch.setattr(threading, "excepthook", _fallback)
    _install_queue_listener_eof_suppressor()
    hook = threading.excepthook
    assert hook is not _fallback, (
        "_install_queue_listener_eof_suppressor did not replace threading.excepthook"
    )
    return hook, recorded


def _make_args(exc_type, *, thread_name: str) -> threading.ExceptHookArgs:
    fake_thread = types.SimpleNamespace(name=thread_name)
    try:
        raise exc_type()
    except exc_type as e:
        exc_value = e
        exc_traceback = e.__traceback__
    return threading.ExceptHookArgs((exc_type, exc_value, exc_traceback, fake_thread))


def test_eof_in_monitor_thread_is_swallowed(installed_hook):
    hook, recorded = installed_hook
    hook(_make_args(EOFError, thread_name="Thread-2 (_monitor)"))
    assert recorded == [], (
        f"Expected EOFError from _monitor thread to be swallowed; "
        f"fallback was invoked with: {recorded!r}"
    )


def test_eof_in_non_monitor_thread_still_surfaces(installed_hook):
    hook, recorded = installed_hook
    hook(_make_args(EOFError, thread_name="some-other-thread"))
    assert len(recorded) == 1, (
        f"Expected one fallback invocation, got {len(recorded)}: {recorded!r}"
    )
    assert recorded[0].exc_type is EOFError
    assert recorded[0].thread.name == "some-other-thread"


def test_non_eof_in_monitor_thread_still_surfaces(installed_hook):
    hook, recorded = installed_hook
    hook(_make_args(RuntimeError, thread_name="Thread-7 (_monitor)"))
    assert len(recorded) == 1, (
        f"Expected one fallback invocation, got {len(recorded)}: {recorded!r}"
    )
    assert recorded[0].exc_type is RuntimeError
    assert recorded[0].thread.name == "Thread-7 (_monitor)"


def test_eof_when_thread_is_none_still_surfaces(installed_hook):
    """Defensive: `args.thread is None` is possible per the typeshed
    annotation. The suppressor must not blow up and must fall through.
    """
    hook, recorded = installed_hook
    fake_args = threading.ExceptHookArgs((EOFError, EOFError(), None, None))
    hook(fake_args)
    assert len(recorded) == 1
    assert recorded[0].exc_type is EOFError
    assert recorded[0].thread is None
