#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-${PWD}/.prometheus_env}"

echo "Installing LeptonInjector (macOS, vendored icecube/LeptonInjector@d203189b)..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../resources/LeptonInjector" && pwd)"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: vendored LeptonInjector source not found at $SRC_DIR" >&2
    exit 1
fi

# -------------------------------
# Build photospline
# (not in Homebrew; must be built from source)
# Install into the venv so no sudo is needed and the dylib is on the RPATH.
# -------------------------------
echo "Building photospline..."
PHOTOSPLINE_TMP=$(mktemp -d)
git clone https://github.com/icecube/photospline.git "$PHOTOSPLINE_TMP/src"
cmake -S "$PHOTOSPLINE_TMP/src" -B "$PHOTOSPLINE_TMP/build" \
    -DCMAKE_INSTALL_PREFIX="$ENV_DIR"
cmake --build "$PHOTOSPLINE_TMP/build" -j"$(sysctl -n hw.logicalcpu)"
cmake --install "$PHOTOSPLINE_TMP/build"
rm -rf "$PHOTOSPLINE_TMP"

# -------------------------------
# Build LeptonInjector
# -------------------------------
echo "Building LeptonInjector from $SRC_DIR..."

# Locate the Homebrew Python that owns this venv.
# We must use its libpython so that boost-python3 (also Homebrew-built) shares
# the same dynamically-linked Python runtime — conda/framework Pythons embed
# their runtime statically and cause a two-runtime segfault on import.
PYTHON_PREFIX=$(python -c "import sys; print(sys.base_prefix)")
PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_LIB=$(find "$PYTHON_PREFIX" -name "libpython${PYTHON_VER}.dylib" | head -1)
PYTHON_INC=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")

if [ -z "$PYTHON_LIB" ]; then
    echo "ERROR: could not find libpython${PYTHON_VER}.dylib under $PYTHON_PREFIX" >&2
    exit 1
fi

echo "  Python prefix : $PYTHON_PREFIX"
echo "  Python lib    : $PYTHON_LIB"

TMP_DIR=$(mktemp -d)
mkdir -p "$TMP_DIR/build"
cd "$TMP_DIR/build"

NJOBS=$(sysctl -n hw.logicalcpu)

cmake \
    -DCMAKE_INSTALL_PREFIX="$ENV_DIR" \
    -DCMAKE_PREFIX_PATH="$ENV_DIR;$(brew --prefix)" \
    -DPython_EXECUTABLE="$ENV_DIR/bin/python" \
    -DPYTHON_EXECUTABLE="$ENV_DIR/bin/python" \
    -DPYTHON_LIBRARY="$PYTHON_LIB" \
    -DPYTHON_INCLUDE_DIR="$PYTHON_INC" \
    -DCMAKE_INSTALL_RPATH="$ENV_DIR/lib;$(brew --prefix)/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    "$SRC_DIR"

make -j"$NJOBS"
make install

# -------------------------------
# Test
# -------------------------------
python - <<PY
import LeptonInjector
print("LeptonInjector OK (macOS)")
PY

rm -rf "$TMP_DIR"
