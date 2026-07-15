#!/usr/bin/env bash
set -euo pipefail

# Build the ppc photon-propagation engine from the icecube/ppc submodule.
#
# The engine source is vendored as a git submodule pinned to a specific
# icecube/ppc commit (see .gitmodules). We build the CPU binary always (needs
# only a C++ compiler) and the GPU/CUDA binary when nvcc is available, placing
# both under PPC_executables/bin/ so a shared filesystem can serve GPU and
# non-GPU machines from the same checkout:
#   bin/ppc_cpu  <- make cpu   (used by the `ppc` config)
#   bin/ppc_gpu  <- make gpu   (used by the `ppc_cuda` config)

SRC="resources/PPC_executables/ppc-src/gpu"
OUT="resources/PPC_executables/bin"

if [ ! -d "$SRC" ]; then
  echo "ppc submodule not found at $SRC - run 'git submodule update --init --recursive' first; skipping"
  exit 0
fi

mkdir -p "$OUT"

echo "Building ppc (CPU)..."
( cd "$SRC" && make clean >/dev/null 2>&1 || true; make cpu ) \
  && cp "$SRC/ppc" "$OUT/ppc_cpu" \
  && echo "  -> $OUT/ppc_cpu" \
  || echo "⚠️ CPU ppc build failed (optional)"

if command -v nvcc >/dev/null 2>&1; then
  echo "Building ppc (GPU)..."
  ( cd "$SRC" && make clean >/dev/null 2>&1 || true; make gpu ) \
    && cp "$SRC/ppc" "$OUT/ppc_gpu" \
    && echo "  -> $OUT/ppc_gpu" \
    || echo "⚠️ GPU ppc build failed (optional)"
else
  echo "nvcc not found; skipping GPU build (build on a CUDA host to populate $OUT/ppc_gpu)"
fi
