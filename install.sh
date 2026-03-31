#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="prometheus"
INSTALL_DIR="${PWD}/.prometheus_env"
USE_MAMBA=1

echo "==== Prometheus Installer ===="

# -------------------------------
# Step 1: Get micromamba (fast, no admin needed)
# -------------------------------
if ! command -v micromamba &> /dev/null; then
    echo "[1/5] Installing micromamba..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj bin/micromamba
    export PATH="$PWD/bin:$PATH"
else
    echo "[1/5] micromamba already available"
fi

# -------------------------------
# Step 2: Create environment
# -------------------------------
echo "[2/5] Creating environment..."
micromamba create -y -p "$INSTALL_DIR" -f environment.yml

# Activate
eval "$(micromamba shell hook -s bash)"
micromamba activate "$INSTALL_DIR"

# -------------------------------
# Step 3: Install PROPOSAL
# -------------------------------
echo "[3/5] Installing PROPOSAL..."
pip install --no-cache-dir proposal

python - <<PY
import proposal
print("PROPOSAL OK:", getattr(proposal, "__version__", "unknown"))
PY

# -------------------------------
# Step 4: Install LeptonInjector
# -------------------------------
echo "[4/5] Installing LeptonInjector..."

echo "Setting up build environment for LeptonInjector..."
# Ensure CONDA_PREFIX is set (micromamba/mamba set this on activation)
: "${CONDA_PREFIX:=$INSTALL_DIR}"
export CONDA_PREFIX
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export HDF5_ROOT="$CONDA_PREFIX"
export BOOST_ROOT="$CONDA_PREFIX"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_ARGS="-DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"

echo "Installing build-time Python helpers (pybind11)..."
pip install --no-cache-dir pybind11

echo "==== DEBUG INFO ===="
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
which cmake || true
cmake --version || true

echo "Attempting to pip-install LeptonInjector (with branch fallback)..."
if ! pip install --no-cache-dir --verbose \
    "git+https://github.com/icecube/LeptonInjector.git@with_earth_py"; then
        echo "Branch install failed; trying master branch..."
        pip install --no-cache-dir --verbose \
            "git+https://github.com/icecube/LeptonInjector.git@master"
fi

python - <<PY
import LeptonInjector
print("LeptonInjector OK")
PY

# -------------------------------
# Step 5: Optional PPC
# -------------------------------
if [ -d "resources/PPC_executables/PPC" ]; then
    echo "[5/5] Building PPC (optional)..."
    pushd resources/PPC_executables/PPC
    make cpu || echo "⚠️ PPC build failed (optional)"
    popd
else
    echo "[5/5] PPC not found, skipping"
fi

# -------------------------------
# Final test
# -------------------------------
echo "Running final test..."

python - <<PY
import numpy
import jax
import proposal
import LeptonInjector
print("All core dependencies installed successfully!")
PY

# Install the local prometheus package into the environment so examples import correctly
echo "Installing local prometheus package into environment..."
pip install -e .


echo ""
echo "==== INSTALL COMPLETE ===="
echo "Activate with:"
echo "  micromamba activate $INSTALL_DIR"