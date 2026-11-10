#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v uv >/dev/null; then
  echo 'uv is required: https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.cache/uv/python}"
case "${1:-core}" in
  core) uv sync --frozen ;;
  libero)
    uv sync --frozen --extra libero
    .venv/bin/python scripts/bootstrap_upstreams.py --only LIBERO
    ;;
  *) echo 'Usage: bash scripts/setup.sh [core|libero]' >&2; exit 2 ;;
esac
