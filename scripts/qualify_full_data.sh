#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 DATA_DIR RUN_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
TASK_UV_CACHE_DIR=${UV_CACHE_DIR:-"$PROJECT_ROOT/.uv-cache"}
export UV_CACHE_DIR="$TASK_UV_CACHE_DIR"

cd "$PROJECT_ROOT"
exec uv run --locked kuairand-agent qualify --data-dir "$1" --run-dir "$2"
