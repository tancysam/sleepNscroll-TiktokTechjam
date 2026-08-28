"""Main-thread signal bridge for cooperative, evidence-preserving cancellation.

The installed POSIX handlers deliberately do one thing only: set a caller-owned
``threading.Event``.  Process supervision, journal writes, cleanup, and state transitions remain
in ordinary Python control flow after the handler returns.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import FrameType
from typing import Any, Final

_DEFAULT_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)
type _SignalHandler = signal.Handlers | Callable[[int, FrameType | None], Any] | int | None


class SignalCancellationError(RuntimeError):
    """A safe main-thread cancellation bridge cannot be installed."""


def _event(value: threading.Event | None) -> threading.Event:
    if value is None:
        return threading.Event()
    if not isinstance(value, threading.Event):
        raise SignalCancellationError("cancel_event must be threading.Event or None")
    return value


def _signals(values: Sequence[signal.Signals | int]) -> tuple[signal.Signals, ...]:
    try:
        normalized = tuple(signal.Signals(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise SignalCancellationError("handled signals must be valid signal numbers") from exc
    if not normalized or len(normalized) != len(set(normalized)):
        raise SignalCancellationError("handled signals must be non-empty and distinct")
    forbidden = {signal.SIGKILL, signal.SIGSTOP}
    if any(item in forbidden for item in normalized):
        raise SignalCancellationError("uncatchable signals cannot request cooperative cancellation")
    return normalized


@contextmanager
def cancellation_on_signals(
    cancel_event: threading.Event | None = None,
    *,
    handled_signals: Sequence[signal.Signals | int] = _DEFAULT_SIGNALS,
) -> Iterator[threading.Event]:
    """Yield an event set by SIGINT/SIGTERM and restore exact prior handlers on exit.

    Python permits signal-handler installation only on the main thread.  Enforcing that boundary
    here produces a deterministic error before any handler changes.  Partial installation is
    rolled back if the platform rejects one requested signal.
    """

    cancellation = _event(cancel_event)
    if threading.current_thread() is not threading.main_thread():
        raise SignalCancellationError("signal cancellation must be installed on the main thread")
    selected = _signals(handled_signals)
    prior: list[tuple[signal.Signals, _SignalHandler]] = []

    def request_cancellation(_signum: int, _frame: FrameType | None) -> None:
        cancellation.set()

    try:
        for item in selected:
            previous = signal.getsignal(item)
            signal.signal(item, request_cancellation)
            prior.append((item, previous))
    except (OSError, RuntimeError, ValueError) as exc:
        for item, previous in reversed(prior):
            signal.signal(item, previous)
        raise SignalCancellationError("could not install cancellation signal handlers") from exc
    try:
        yield cancellation
    finally:
        for item, previous in reversed(prior):
            signal.signal(item, previous)


__all__ = [
    "SignalCancellationError",
    "cancellation_on_signals",
]
