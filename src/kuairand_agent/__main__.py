"""Module entry point for ``python -m kuairand_agent``."""

from __future__ import annotations

from kuairand_agent.cli import entrypoint

if __name__ == "__main__":  # pragma: no cover - exercised by subprocess smoke tests
    entrypoint()
