#!/usr/bin/env bash
set -euo pipefail

echo "Installing LeptonInjector (legacy branch)..."

# -------------------------------
# Build environment
# -------------------------------
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export HDF5_ROOT="$CONDA_PREFIX"
export BOOST_ROOT="$CONDA_PREFIX"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"

# -------------------------------
# Clone
# -------------------------------
TMP_DIR=$(mktemp -d)
echo "Cloning into $TMP_DIR"

git clone -b with_earth_py https://github.com/icecube/LeptonInjector.git "$TMP_DIR/src"

# -------------------------------
# Patch LeptonInjector sources
# (fixes GCC 13 / SuiteSparse 7 compatibility)
# -------------------------------
echo "Applying compatibility patches to LeptonInjector sources..."

# Patch 1: SuiteSparse.cmake – STRING(REGEX REPLACE) on the whole file content
# leaves the entire header as the version variable, breaking IF(version GREATER 3).
# Use STRING(REGEX MATCH) first so we only pass the matched line to REGEX REPLACE.
_SS_CMAKE="$TMP_DIR/src/cmake/Packages/SuiteSparse.cmake"
_PATCH_PY=$(mktemp --suffix=.py)
cat > "$_PATCH_PY" << 'PYEOF'
import sys, pathlib
f = pathlib.Path(sys.argv[1])
txt = f.read_text()
old = (
    '    STRING (REGEX REPLACE ".*define SUITESPARSE_MAIN_VERSION ([0-9]+).*" "\\\\1"\n'
    '      SUITESPARSE_VERSION "${_SUITESPARSE_VERSION}")'
)
new = (
    '    STRING (REGEX MATCH "define SUITESPARSE_MAIN_VERSION +([0-9]+)" _SS_M "${_SUITESPARSE_VERSION}")\n'
    '    IF (_SS_M)\n'
    '      STRING (REGEX REPLACE ".*([0-9]+)$" "\\\\1" SUITESPARSE_VERSION "${_SS_M}")\n'
    '    ELSE ()\n'
    '      SET (SUITESPARSE_VERSION 4)\n'
    '    ENDIF ()'
)
if old in txt:
    f.write_text(txt.replace(old, new))
    print("  * SuiteSparse.cmake patched OK")
else:
    print("  * SuiteSparse.cmake: expected pattern not found, skipping", file=sys.stderr)
PYEOF
python3 "$_PATCH_PY" "$_SS_CMAKE"
rm -f "$_PATCH_PY"

# Patch 2: Coordinates.h – uint8_t requires <cstdint> (GCC 13 no longer includes it
# transitively through <array> or other standard headers).
_COORDS_H="$TMP_DIR/src/public/LeptonInjector/Coordinates.h"
if ! grep -q '<cstdint>' "$_COORDS_H"; then
  sed -i 's|#include <array>|#include <array>\n#include <cstdint>|' "$_COORDS_H"
  echo "  * Coordinates.h: added #include <cstdint>"
fi

# Patch 3: Particle.h – uint8_t requires <cstdint>
_PARTICLE_H="$TMP_DIR/src/public/LeptonInjector/Particle.h"
if ! grep -q '<cstdint>' "$_PARTICLE_H"; then
  sed -i 's|#include <LeptonInjector/Constants\.h>|#include <LeptonInjector/Constants.h>\n#include <cstdint>|' "$_PARTICLE_H"
  echo "  * Particle.h: added #include <cstdint>"
fi

# Patch 4: Particle.cxx – std::runtime_error requires <stdexcept>
_PARTICLE_CXX="$TMP_DIR/src/private/LeptonInjector/Particle.cxx"
if ! grep -q '<stdexcept>' "$_PARTICLE_CXX"; then
  sed -i 's|#include <math\.h>|#include <math.h>\n#include <stdexcept>|' "$_PARTICLE_CXX"
  echo "  * Particle.cxx: added #include <stdexcept>"
fi

# -------------------------------
# Build
# -------------------------------
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
  ../src

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