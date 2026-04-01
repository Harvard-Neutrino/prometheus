#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$1"

# Detect OS and architecture so we download the right micromamba binary.
_OS="$(uname -s)"
_ARCH="$(uname -m)"
case "${_OS}-${_ARCH}" in
    Linux-x86_64)   MAMBA_PLATFORM="linux-64" ;;
    Linux-aarch64)  MAMBA_PLATFORM="linux-aarch64" ;;
    Darwin-x86_64)  MAMBA_PLATFORM="osx-64" ;;
    Darwin-arm64)   MAMBA_PLATFORM="osx-arm64" ;;
    *)
        echo "Unsupported platform: ${_OS}-${_ARCH}" >&2
        exit 1
        ;;
esac

# Download a fresh micromamba if the one in bin/ is absent or not executable
# on this platform (e.g. a Linux binary left over on a Mac checkout).
if ! "${PWD}/bin/micromamba" --version &>/dev/null; then
    echo "Downloading micromamba for ${MAMBA_PLATFORM}..."
    mkdir -p "${PWD}/bin"
    curl -Ls "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" \
        | tar -xvj -C "${PWD}" bin/micromamba
fi
export PATH="$PWD/bin:$PATH"

echo "Creating environment at $ENV_DIR"
micromamba create -y -p "$ENV_DIR" -f environment.yml