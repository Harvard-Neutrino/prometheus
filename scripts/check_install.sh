#!/usr/bin/env bash
set -euo pipefail

echo "Running installation checks..."

python - <<PY
import numpy
import jax
import proposal
import LeptonInjector

print("All imports successful!")
PY

echo "✔ Installation verified"