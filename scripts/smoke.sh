#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
TASK_UV_CACHE_DIR=${UV_CACHE_DIR:-"$PROJECT_ROOT/.uv-cache"}
export UV_CACHE_DIR="$TASK_UV_CACHE_DIR"

cd "$PROJECT_ROOT"
uv sync --locked --offline --no-group research-tree --no-group research-neural
uv run --locked --offline --no-group research-tree --no-group research-neural kuairand-agent --help >/dev/null
uv run --locked --offline --no-group research-tree --no-group research-neural pytest -q tests/unit
