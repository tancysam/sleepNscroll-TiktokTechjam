#!/bin/sh
set -eu

if [ "$#" -gt 3 ]; then
  echo "usage: $0 [CONFIG_PATH [QUALIFICATION_RUN_DIR [RUN_DIR]]]" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
TASK_UV_CACHE_DIR=${UV_CACHE_DIR:-"$PROJECT_ROOT/.uv-cache"}
export UV_CACHE_DIR="$TASK_UV_CACHE_DIR"
CONFIG_PATH=${1:-"$PROJECT_ROOT/configs/full-pure.toml"}
QUALIFICATION_RUN_DIR=${2:-"$PROJECT_ROOT/runs/wp3-official-qualification"}
RUN_DIR=${3:-}

cd "$PROJECT_ROOT"
if [ -f "$PROJECT_ROOT/.env.local" ]; then
  set -a
  . "$PROJECT_ROOT/.env.local"
  set +a
fi
uv sync --locked --group research-tree --no-group research-neural
if [ -n "$RUN_DIR" ]; then
  exec uv run --locked --group research-tree --no-group research-neural kuairand-agent run \
    --config "$CONFIG_PATH" \
    --qualification-run-dir "$QUALIFICATION_RUN_DIR" \
    --run-dir "$RUN_DIR"
fi
exec uv run --locked --group research-tree --no-group research-neural kuairand-agent run \
  --config "$CONFIG_PATH" \
  --qualification-run-dir "$QUALIFICATION_RUN_DIR"
