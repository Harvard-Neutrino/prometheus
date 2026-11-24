#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPC_DIR="$ROOT/resources/PPC_executables/PPC"
OFFICIAL_REL="resources/PPC_executables/icecube-ppc"
OFFICIAL_DIR="$ROOT/$OFFICIAL_REL"

if [ -d "$PPC_DIR" ]; then
  echo "Building bundled PPC (legacy modules only)..."
  (cd "$PPC_DIR" && make cpu) || echo "WARNING: bundled PPC build failed (optional)"
else
  echo "Bundled PPC directory not found, skipping"
fi

# Official IceCube PPC, pinned as a git submodule. Needed for multi-PMT modules.
# Only initialise when empty, so an intentionally changed checkout is kept.
if [ ! -f "$OFFICIAL_DIR/gpu/Makefile" ]; then
  git -C "$ROOT" submodule update --init "$OFFICIAL_REL" || true
fi

if [ -f "$OFFICIAL_DIR/gpu/Makefile" ]; then
  echo "Building official PPC at $(git -C "$OFFICIAL_DIR" rev-parse --short HEAD)..."
  # The upstream Makefile passes --fast-math, which clang rejects.
  # -ffast-math is the same option for both gcc and clang.
  (cd "$OFFICIAL_DIR/gpu" && make cpu gcpp="${CXX:-c++} -x c++ ppc.cu -O2 -ffast-math") \
    || echo "WARNING: official PPC build failed (optional)"
else
  echo "WARNING: official PPC source not available, skipping (optional)"
fi
