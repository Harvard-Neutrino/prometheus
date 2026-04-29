#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$1"

_OS="$(uname -s)"
_ARCH="$(uname -m)"

if [ "$_OS" = "Darwin" ]; then
    # macOS: Homebrew + plain Python venv.
    # Homebrew Python is required so that boost-python3 (also Homebrew-built)
    # shares the same dynamically-linked Python runtime. conda and
    # actions/setup-python both statically embed their runtime, which causes a
    # two-runtime segfault when importing compiled extensions.
    if ! command -v brew >/dev/null 2>&1; then
        echo "ERROR: Homebrew is required. Install it from https://brew.sh then re-run." >&2
        exit 1
    fi

    echo "Installing Homebrew dependencies..."
    brew install cmake hdf5 boost boost-python3 eigen gsl pkg-config cfitsio

    BREW_PYTHON="$(brew --prefix)/bin/python3"
    if [ ! -x "$BREW_PYTHON" ]; then
        echo "ERROR: Homebrew Python not found at $BREW_PYTHON." >&2
        echo "Install it with: brew install python3" >&2
        exit 1
    fi

    echo "Creating Python environment at $ENV_DIR..."
    "$BREW_PYTHON" -m venv "$ENV_DIR"
    "$ENV_DIR/bin/pip" install --upgrade pip
    "$ENV_DIR/bin/pip" install numpy scipy matplotlib cython pybind11 jax jaxlib
else
    # Linux/WSL2: micromamba environment from environment.yml.
    case "${_OS}-${_ARCH}" in
        Linux-x86_64)   MAMBA_PLATFORM="linux-64" ;;
        Linux-aarch64)  MAMBA_PLATFORM="linux-aarch64" ;;
        *)
            echo "Unsupported platform: ${_OS}-${_ARCH}" >&2
            echo "Supported: Linux x86-64, Linux aarch64, macOS arm64." >&2
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
fi
