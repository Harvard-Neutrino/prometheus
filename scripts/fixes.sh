#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/fixes.sh [ENV_DIR]
ENV_DIR="${1:-${PWD}/.prometheus_env}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Running post-install fixes in: $ENV_DIR"

PY="$ENV_DIR/bin/python3"

if [ ! -x "$PY" ]; then
  echo "Error: python not found in $ENV_DIR" >&2
  exit 2
fi

echo "- Installing prometheus (editable) into environment..."
"$PY" -m pip install -e "$REPO_ROOT"

echo "- Installing fennel-seed v2.1.0 from GitHub..."
"$PY" -m pip install "fennel_seed @ https://github.com/MeighenBergerS/fennel/archive/refs/tags/v2.1.0.tar.gz"

echo "- Installing dev tools (pytest, ruff)..."
"$PY" -m pip install pytest ruff

echo "Post-install fixes complete."
