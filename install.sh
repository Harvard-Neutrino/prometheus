#!/usr/bin/env bash
set -euo pipefail

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