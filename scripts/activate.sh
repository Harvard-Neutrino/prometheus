#!/usr/bin/env bash

ENV_DIR="$1"

# Normalize to absolute path without relying on any external binary.
# realpath -m (GNU) does not exist on macOS; dirname/cd works everywhere.
case "$ENV_DIR" in
	/*) ;;
	*) ENV_DIR="$PWD/$ENV_DIR" ;;
esac

# Plain Python venv (macOS installer) — activate directly.
if [ -f "$ENV_DIR/bin/activate" ]; then
	source "$ENV_DIR/bin/activate"
	return 0 2>/dev/null || exit 0
fi

# micromamba environment (Linux/WSL2 installer).
if command -v micromamba >/dev/null 2>&1; then
	MICO=micromamba
else
	# Use BASH_SOURCE so the script works when sourced
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
	if [ -x "$REPO_ROOT/bin/micromamba" ]; then
		MICO="$REPO_ROOT/bin/micromamba"
	else
		echo "micromamba not found; please install micromamba or add it to PATH" >&2
		exit 1
	fi
fi

# The micromamba shell hook may reference variables not yet defined
# (e.g. MAMBA_ROOT_PREFIX). Temporarily disable 'nounset' while evaluating
# it, then restore. Detect the calling shell (bash or zsh).
_SHELL_NAME="$(basename "${SHELL:-bash}")"
case "$_SHELL_NAME" in
	zsh|bash) ;;
	*) _SHELL_NAME=bash ;;
esac

set +u
eval "$("$MICO" shell hook -s "$_SHELL_NAME")"
set -u

micromamba activate -p "$ENV_DIR"