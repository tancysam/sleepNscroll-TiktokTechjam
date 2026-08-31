#!/bin/sh
set -eu

if [ "${KUAIRAND_ENABLE_GPU_QUALIFICATION:-}" != "1" ]; then
  echo "GPU qualification is opt-in; set KUAIRAND_ENABLE_GPU_QUALIFICATION=1" >&2
  exit 2
fi

if [ "$#" -ne 0 ]; then
  echo "usage: KUAIRAND_ENABLE_GPU_QUALIFICATION=1 $0" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
TASK_UV_CACHE_DIR=${UV_CACHE_DIR:-"$PROJECT_ROOT/.uv-cache"}
export UV_CACHE_DIR="$TASK_UV_CACHE_DIR"

# tree-gpu is intentionally not provisioned from the stock PyPI LightGBM wheel. This
# no-sync probe inspects the explicitly prepared local environment and qualifies it only
# after real GPU training plus a same-backend deterministic replay.
cd "$PROJECT_ROOT"
exec uv run --locked --no-sync \
  --group tree-gpu \
  --no-group dev \
  --no-group research-tree \
  --no-group research-neural \
  --no-group tree-cpu \
  python -m kuairand_agent.resource_profiles qualify-gpu configs/competition-gpu.toml
