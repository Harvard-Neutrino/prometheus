#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$1"

# Detect OS and architecture so we download the right micromamba binary.
# Only Linux is supported; macOS users should use the Docker image.
_OS="$(uname -s)"
_ARCH="$(uname -m)"
case "${_OS}-${_ARCH}" in
    Linux-x86_64)   MAMBA_PLATFORM="linux-64" ;;
    Linux-aarch64)  MAMBA_PLATFORM="linux-aarch64" ;;
    *)
        echo "Unsupported platform: ${_OS}-${_ARCH}" >&2
        echo "Only Linux (x86-64, aarch64) is supported. macOS users should use the Docker image." >&2
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