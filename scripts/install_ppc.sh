#!/usr/bin/env bash
set -euo pipefail

PPC_DIR="resources/PPC_executables/PPC"

if [ ! -d "$PPC_DIR" ]; then
  echo "PPC directory not found, skipping"
  exit 0
fi

echo "Building PPC..."
cd "$PPC_DIR"

make cpu || echo "⚠️ PPC build failed (optional)"