from __future__ import annotations

import signal
import threading

import pytest

from kuairand_agent.execution.signals import (
    SignalCancellationError,
    cancellation_on_signals,
)


def test_signal_context_sets_supplied_event_and_restores_exact_prior_handlers() -> None:
    cancellation = threading.Event()
    prior = {item: signal.getsignal(item) for item in (signal.SIGINT, signal.SIGTERM)}

    with cancellation_on_signals(cancellation) as installed:
        assert installed is cancellation
        assert not cancellation.is_set()
        active = signal.getsignal(signal.SIGTERM)
        assert callable(active)
        active(signal.SIGTERM, None)
        assert cancellation.is_set()

    assert {item: signal.getsignal(item) for item in prior} == prior


def test_signal_context_rejects_installation_outside_the_main_thread() -> None:
    observed: list[BaseException] = []

    def install() -> None:
        try:
            with cancellation_on_signals():
                pass
        except BaseException as exc:
            observed.append(exc)

    thread = threading.Thread(target=install)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(observed) == 1
    assert isinstance(observed[0], SignalCancellationError)
    assert "main thread" in str(observed[0])


def test_signal_context_rejects_invalid_event_without_changing_handlers() -> None:
    prior = signal.getsignal(signal.SIGINT)

    with (
        pytest.raises(SignalCancellationError, match=r"threading\.Event"),
        cancellation_on_signals(object()),  # type: ignore[arg-type]
    ):
        pass

    assert signal.getsignal(signal.SIGINT) is prior
