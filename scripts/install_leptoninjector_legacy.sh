#!/usr/bin/env bash
set -euo pipefail

echo "Installing LeptonInjector (vendored, icecube/LeptonInjector@d203189b)..."

# Locate the vendored source relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../resources/LeptonInjector" && pwd)"

if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: vendored LeptonInjector source not found at $SRC_DIR" >&2
  exit 1
fi

echo "Building from $SRC_DIR"

# -------------------------------
# Build environment
# -------------------------------
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export HDF5_ROOT="$CONDA_PREFIX"
export BOOST_ROOT="$CONDA_PREFIX"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"

# Compatibility patches are already baked into the vendored source.

# -------------------------------
# Build
# -------------------------------
TMP_DIR=$(mktemp -d)
mkdir -p "$TMP_DIR/build"
cd "$TMP_DIR/build"

# Ensure necessary runtime/build deps are present in the env similar to the Dockerfile
echo "Ensuring photospline and CMake 3.22 are available in environment ($CONDA_PREFIX)..."
if command -v micromamba >/dev/null 2>&1; then
  micromamba install -y -p "$CONDA_PREFIX" -c conda-forge photospline cmake=3.22
elif [ -x "$CONDA_PREFIX/bin/conda" ]; then
  "$CONDA_PREFIX/bin/conda" install -y -p "$CONDA_PREFIX" -c conda-forge photospline cmake=3.22
else
  echo "Warning: micromamba/conda not available; proceeding with system cmake (may fail)" >&2
fi

# Prefer the CMake shipped into the environment if present

# Prefer the CMake shipped into the environment if present and recent enough,
# otherwise build a local CMake 3.22.2 (matches container/new_kernel/dockerfile).
need_cmake=3.22
use_cmake=""
if command -v cmake >/dev/null 2>&1; then
  ver_str=$(cmake --version | head -n1 | awk '{print $3}')
  major=$(echo "$ver_str" | cut -d. -f1)
  minor=$(echo "$ver_str" | cut -d. -f2)
  if [ "$major" -gt 3 ] || ( [ "$major" -eq 3 ] && [ "$minor" -ge 22 ] ); then
    use_cmake="$(command -v cmake)"
  fi
fi

if [ -z "$use_cmake" ]; then
  echo "Building CMake ${need_cmake} locally (this may take a few minutes)..."
  cd "$TMP_DIR"
  wget -q https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2.tar.gz
  tar -zxvf cmake-3.22.2.tar.gz
  cd cmake-3.22.2
  ./bootstrap >/dev/null
  make -j$(nproc) >/dev/null
  CMAKE_BIN="$PWD/bin/cmake"
  cd "$TMP_DIR/build"
else
  CMAKE_BIN="$use_cmake"
fi

"$CMAKE_BIN" \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  "$SRC_DIR"

make -j$(nproc)
make install

# -------------------------------
# Python path fix
# -------------------------------
PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

export PYTHONPATH="$CONDA_PREFIX/lib/python${PYVER}/site-packages:${PYTHONPATH:-}"

# -------------------------------
# Test
# -------------------------------
python - <<PY
import LeptonInjector
print("LeptonInjector OK (legacy)")
PY

# Cleanup
rm -rf "$TMP_DIR"