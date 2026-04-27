#!/usr/bin/env bash
# scripts/docker_test.sh — build and smoke-test the Prometheus Docker image locally.
#
# Usage:
#   bash scripts/docker_test.sh [--gpu] [--tag TAG] [--no-build]
#
#   --gpu        Build and test the GPU image (requires nvidia/cuda base + nvcc).
#   --tag TAG    Override the image tag (default: prometheus:cpu or prometheus:gpu).
#   --no-build   Skip the build step (test an already-built image).
#   --e2e        Also run the 100-event physics regression test inside the container
#                (adds ~8 minutes; skipped by default).
#
# Exit codes:
#   0  All tests passed.
#   1  Build failed.
#   2  Smoke tests failed.
#   3  Fast unit tests failed.
#   4  E2E test failed (only when --e2e is passed).
#
# The script runs entirely offline once the image is built.
# The build requires internet access for micromamba and pip packages.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
GPU=0
TAG=""
NO_BUILD=0
E2E=0

# ─── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPU=1       ; shift ;;
        --tag)      TAG="$2"    ; shift 2 ;;
        --no-build) NO_BUILD=1  ; shift ;;
        --e2e)      E2E=1       ; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TAG" ]]; then
    if [[ "$GPU" -eq 1 ]]; then
        TAG="prometheus:gpu"
    else
        TAG="prometheus:cpu"
    fi
fi

DOCKERFILE="container/Dockerfile"
if [[ "$GPU" -eq 1 ]]; then
    DOCKERFILE="container/Dockerfile.gpu"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ─── Step 1: Build ───────────────────────────────────────────────────────────
if [[ "$NO_BUILD" -eq 0 ]]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo " Building image: $TAG"
    echo " Dockerfile:     $DOCKERFILE"
    echo "════════════════════════════════════════════"
    if ! docker build -f "$DOCKERFILE" -t "$TAG" .; then
        echo ""
        echo "ERROR: Docker build failed." >&2
        exit 1
    fi
    echo "Build succeeded: $TAG"
fi

# ─── Step 2: Smoke test — import all live submodules ────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo " Smoke test: import checks"
echo "════════════════════════════════════════════"
SMOKE_CMD='cd /opt/prometheus && python -m pytest tests/test_smoke.py --tb=short -q'
if ! docker run --rm "$TAG" bash -c "$SMOKE_CMD"; then
    echo ""
    echo "ERROR: Smoke tests failed." >&2
    exit 2
fi

# ─── Step 3: Fast unit tests ─────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo " Unit tests (fast)"
echo "════════════════════════════════════════════"
FAST_CMD='cd /opt/prometheus && python -m pytest --ignore=tests/test_e2e.py --tb=short -q'
if ! docker run --rm "$TAG" bash -c "$FAST_CMD"; then
    echo ""
    echo "ERROR: Fast unit tests failed." >&2
    exit 3
fi

# ─── Step 4 (optional): E2E physics regression test ─────────────────────────
if [[ "$E2E" -eq 1 ]]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo " E2E test: 100-event water simulation"
    echo "════════════════════════════════════════════"
    E2E_CMD='cd /opt/prometheus && python -m pytest tests/test_e2e.py --run-slow --tb=short -q'
    if ! docker run --rm "$TAG" bash -c "$E2E_CMD"; then
        echo ""
        echo "ERROR: E2E test failed." >&2
        exit 4
    fi
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo " All tests passed for image: $TAG"
if [[ "$E2E" -eq 0 ]]; then
    echo " (E2E test skipped — run with --e2e to include it)"
fi
echo "════════════════════════════════════════════"
echo ""
