#!/usr/bin/env bash

ENV_DIR="$1"

# Normalize to absolute path (works even if the path does not yet exist)
if command -v realpath >/dev/null 2>&1; then
	ENV_DIR="$(realpath -m "$ENV_DIR")"
else
	# Fallback: resolve relative paths; this fails if ENV_DIR doesn't exist.
	case "$ENV_DIR" in
		/*) ;;
		*) ENV_DIR="$PWD/$ENV_DIR" ;;
	esac
fi

# Prefer system micromamba if available, otherwise fall back to repository bin/micromamba
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
# (e.g. MAMBA_ROOT_PREFIX). When this script is sourced from a shell
# with `set -u` enabled, those references cause an immediate failure.
# Temporarily disable 'nounset' while evaluating the hook, then restore it.
set +u
eval "$("$MICO" shell hook -s bash)"
set -u

micromamba activate -p "$ENV_DIR"