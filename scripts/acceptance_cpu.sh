#!/bin/sh
set -eu

if [ "${KUAIRAND_ENABLE_CPU_ACCEPTANCE:-}" != "1" ]; then
  echo "CPU acceptance is opt-in; set KUAIRAND_ENABLE_CPU_ACCEPTANCE=1" >&2
  exit 2
fi

if [ "$#" -ne 0 ]; then
  echo "usage: KUAIRAND_ENABLE_CPU_ACCEPTANCE=1 $0" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
TASK_UV_CACHE_DIR=${UV_CACHE_DIR:-"$PROJECT_ROOT/.uv-cache"}
export UV_CACHE_DIR="$TASK_UV_CACHE_DIR"

cd "$PROJECT_ROOT"
uv sync --locked \
  --group dev \
  --group tree-cpu \
  --no-group research-tree \
  --no-group research-neural \
  --no-group tree-gpu
uv run --locked --no-sync \
  --group dev \
  --group tree-cpu \
  --no-group research-tree \
  --no-group research-neural \
  --no-group tree-gpu \
  python -m kuairand_agent.resource_profiles validate configs/competition-cpu.toml
exec uv run --locked --no-sync \
  --group dev \
  --group tree-cpu \
  --no-group research-tree \
  --no-group research-neural \
  --no-group tree-gpu \
  pytest -q \
  tests/unit/test_resource_profiles.py \
  tests/unit/test_dependency_profiles.py \
  tests/unit/test_tree_ranker.py
