#!/usr/bin/env bash
set -euo pipefail

# macOS is not supported. PROPOSAL's Conan/Boost build and LeptonInjector's
# CMake build both have unresolved failures on macOS. Use the Docker image.
if [ "$(uname -s)" = "Darwin" ]; then
    echo "" >&2
    echo "ERROR: macOS is not supported by this installer." >&2
    echo "" >&2
    echo "Prometheus requires PROPOSAL (Boost via Conan) and LeptonInjector" >&2
    echo "(CMake + HDF5 + Boost), both of which fail to build on macOS." >&2
    echo "" >&2
    echo "Please use the Docker image instead:" >&2
    echo "  https://github.com/Harvard-Neutrino/prometheus#install-with-containers" >&2
    echo "" >&2
    exit 1
fi

ENV_DIR="${PWD}/.prometheus_env"

# Ensure the repo-local micromamba binary (downloaded by setup_env.sh) is
# available to all sub-scripts that are invoked as separate bash processes.
export PATH="${PWD}/bin:${PATH}"

echo "==== Prometheus Installer ===="

# Parse args
WITH_PPC=0
for arg in "$@"; do
  case $arg in
    --with-ppc) WITH_PPC=1 ;;
  esac
done

# Step 1: Setup environment
bash scripts/setup_env.sh "$ENV_DIR"

# Activate
source scripts/activate.sh "$ENV_DIR"

# Step 2: Install core deps
bash scripts/install_proposal.sh
bash scripts/install_leptoninjector_legacy.sh

# Step 3: Optional PPC
if [ "$WITH_PPC" -eq 1 ]; then
    bash scripts/install_ppc.sh
fi

# Step 4: Check install
bash scripts/check_install.sh

# Step 5: Post-install fixes (install editable prometheus and GitHub fennel)
bash scripts/fixes.sh "$ENV_DIR"

echo ""
echo "==== INSTALL COMPLETE ===="
echo "Activate with:"
echo "  source scripts/activate.sh $ENV_DIR"