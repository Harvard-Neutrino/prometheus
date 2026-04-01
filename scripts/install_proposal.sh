#!/usr/bin/env bash
set -euo pipefail

echo "Installing PROPOSAL..."

pip install --no-cache-dir proposal

python - <<PY
import proposal
print("PROPOSAL OK:", getattr(proposal, "__version__", "unknown"))
PY