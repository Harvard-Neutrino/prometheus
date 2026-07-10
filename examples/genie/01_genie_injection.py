#!/usr/bin/env python3
"""05_genie_injection.py
Example showing how to drive Prometheus with GENIE event data.

Instead of running ``LeptonInjector``, this example reads pre-generated GENIE
events from a gRooTracker ROOT file and feeds them into the Prometheus
photon-propagation pipeline.

Two vertex placement modes are supported:

* ``fixed`` — all events are placed at a user-supplied position (or at the
  detector centre when no position is given).  Good for quick studies where
  exact vertex placement does not matter.

* ``random`` — vertices are sampled uniformly inside the detector bounding
  cylinder.  Requires the detector geometry to be fully loaded.

Usage
-----
Run from the repository root::

    /path/to/.prometheus_env/bin/python examples/05_genie_injection.py
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from prometheus import Prometheus, config
except Exception:
    logger.exception(
        "Error importing Prometheus. "
        "Ensure the environment is activated and requirements are installed."
    )
    logger.info("Hint: source scripts/activate.sh .prometheus_env")
    sys.exit(1)

# Prefer CPU JAX to avoid needing a GPU for this example.
try:
    from jax.config import config as jconfig

    jconfig.update("jax_enable_x64", True)
    jconfig.update("jax_platform_name", "cpu")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENIE_FILE = REPO_ROOT / "tests" / "resources" / "genie_example.root"
GEO_FILE = REPO_ROOT / "resources" / "geofiles" / "demo_water.geo"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    config.run.run_number = 1
    config.run.random_state_seed = 1
    config.run.storage_prefix = str(REPO_ROOT / "examples" / "output" / "genie") + "/"

    # --- Detector ---
    config.detector.geo_file = str(GEO_FILE)

    # --- GENIE injection ---
    config.injection.name = "GENIE"

    # Point injection_file at the GENIE ROOT file; Prometheus reads it directly.
    config.injection.genie.paths.injection_file = str(GENIE_FILE)

    # inject=True runs the validator (checks the ROOT file is readable) before
    # the simulation loop.  Set to False to skip validation.
    config.injection.genie.inject = True

    # Placement mode 1 — fixed position.
    # All events are placed at the detector centre when positions is not set.
    # Supply an explicit [x, y, z] list to override.
    config.injection.genie.simulation.placement = "fixed"
    # config.injection.genie.simulation.positions = [x, y, z]  # metres

    # Placement mode 2 — random (uncomment to use instead of fixed above).
    # Vertices are sampled uniformly inside the detector bounding cylinder.
    # config.injection.genie.simulation.placement = "random"
    # config.injection.genie.simulation.positions = None

    print(f"Loading GENIE events from {GENIE_FILE.name}")
    prom = Prometheus()

    try:
        prom.sim()
    except Exception:
        logger.exception("Simulation error during prom.sim()")
        sys.exit(1)

    print(f"Simulation complete — propagated {len(prom.injection)} GENIE events")


if __name__ == "__main__":
    main()
