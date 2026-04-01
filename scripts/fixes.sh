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

echo "- Installing fennel from GitHub (latest recommended branch)..."
# The GitHub repository packages as 'fennel-seed' in its metadata; request that name.
"$PY" -m pip install --upgrade "git+https://github.com/MeighenBergerS/fennel.git@master#egg=fennel-seed"

echo "Post-install fixes complete."
