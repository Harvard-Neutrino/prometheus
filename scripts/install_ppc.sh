#!/usr/bin/env bash
set -euo pipefail

# PPC is an IceCube/Linux tool. It uses GNU make flags and Linux-specific
# compiler options that are not compatible with macOS/Apple Clang.
if [ "$(uname -s)" = "Darwin" ]; then
  echo "PPC is not supported on macOS; skipping."
  exit 0
fi

PPC_DIR="resources/PPC_executables/PPC"

if [ ! -d "$PPC_DIR" ]; then
  echo "PPC directory not found, skipping"
  exit 0
fi

echo "Building PPC..."
cd "$PPC_DIR"

make cpu || echo "⚠️ PPC build failed (optional)"