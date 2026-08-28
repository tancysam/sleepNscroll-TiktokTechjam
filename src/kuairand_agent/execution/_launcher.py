"""Private pre-exec handshake used by :mod:`kuairand_agent.execution.runner`.

The launcher deliberately depends only on the Python standard library.  It starts in the
candidate's new session, blocks until the trusted controller writes the exact execution nonce,
and only then replaces itself with the candidate interpreter.  Closing the control pipe before
release makes the launcher exit without executing candidate code.
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import suppress
from typing import Final

_NONCE_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_MAX_HANDSHAKE_BYTES: Final = 256
_EXIT_USAGE: Final = 120
_EXIT_NOT_RELEASED: Final = 121
_EXIT_PROTOCOL: Final = 122
_EXIT_EXEC: Final = 123


def _read_release(control_fd: int) -> bytes:
    payload = bytearray()
    while len(payload) <= _MAX_HANDSHAKE_BYTES:
        chunk = os.read(control_fd, min(64, _MAX_HANDSHAKE_BYTES + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
        if b"\n" in chunk:
            break
    return bytes(payload)


def main() -> int:
    """Wait for the trusted release token, then ``execve`` the intended candidate command."""

    if len(sys.argv) < 5:
        return _EXIT_USAGE
    try:
        control_fd = int(sys.argv[1])
    except ValueError:
        return _EXIT_USAGE
    nonce = sys.argv[2]
    interpreter = sys.argv[3]
    arguments = sys.argv[4:]
    if (
        control_fd < 0
        or _NONCE_PATTERN.fullmatch(nonce) is None
        or not os.path.isabs(interpreter)
        or not arguments
    ):
        return _EXIT_USAGE

    try:
        payload = _read_release(control_fd)
    except OSError:
        return _EXIT_NOT_RELEASED
    finally:
        with suppress(OSError):
            os.close(control_fd)

    if not payload:
        return _EXIT_NOT_RELEASED
    expected = f"{nonce}\n".encode("ascii")
    if payload != expected or os.environ.get("KUAIRAND_EXECUTION_NONCE") != nonce:
        return _EXIT_PROTOCOL

    try:
        os.execve(interpreter, [interpreter, *arguments], dict(os.environ))
    except OSError:
        return _EXIT_EXEC


if __name__ == "__main__":  # pragma: no cover - exercised through Runner subprocess tests
    raise SystemExit(main())
