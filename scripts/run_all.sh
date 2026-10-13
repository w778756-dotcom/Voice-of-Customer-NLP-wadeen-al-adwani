#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/voice_of_customer-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export OMP_NUM_THREADS=2
uv sync --frozen
uv run --frozen python scripts/train.py
uv run --frozen python scripts/evaluate.py
uv run --frozen pytest -q
