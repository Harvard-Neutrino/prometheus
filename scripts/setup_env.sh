#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$1"

if ! command -v micromamba &> /dev/null; then
    echo "Installing micromamba..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj bin/micromamba
    export PATH="$PWD/bin:$PATH"
fi

echo "Creating environment at $ENV_DIR"
micromamba create -y -p "$ENV_DIR" -f environment.yml