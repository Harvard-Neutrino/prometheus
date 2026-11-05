#!/usr/bin/env bash
set -euo pipefail

echo "Installing PROPOSAL..."

pip install --no-cache-dir proposal

# On Linux the module links libpython but only searches its own directory, so
# it fails to load unless the system happens to provide the same libpython. Add
# the environment's library directory, which is two levels above site-packages.
if [ "$(uname -s)" = "Linux" ]; then
    pip install --no-cache-dir patchelf
    PROPOSAL_MODULE="$(python - <<PY
import glob, os, sysconfig
print(glob.glob(os.path.join(sysconfig.get_path("platlib"), "proposal*.so"))[0])
PY
)"
    patchelf --set-rpath '$ORIGIN:$ORIGIN/../..' "$PROPOSAL_MODULE"
fi

python - <<PY
import proposal
print("PROPOSAL OK:", getattr(proposal, "__version__", "unknown"))
PY
