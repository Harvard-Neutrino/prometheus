#!/usr/bin/env python3
"""02_basic_ice.py
Minimal ice-case example to validate a Prometheus install with PPC.

Runs a single-event CPU-only simulation using the demo ice geo file
and the south-pole PPC ice tables bundled in resources/.
"""
import sys
import traceback

try:
    from prometheus import Prometheus, config
except Exception as e:
    print("Error importing Prometheus:", e)
    print("Ensure you activated the environment:")
    print("  source scripts/activate.sh .prometheus_env")
    sys.exit(1)

# Use CPU-only JAX
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
except Exception:
    pass


def main():
    # Minimal runtime configuration
    config.run.run_number = 2
    config.run.random_state_seed = 2
    config.run.nevents = 3

    # Injection: volume (non-ranged) — vertex placed inside the detector volume
    config.injection.name = "LeptonInjector"
    config.injection.lepton_injector.simulation.is_ranged = False
    config.injection.lepton_injector.simulation.final_state_1 = "MuMinus"
    config.injection.lepton_injector.simulation.final_state_2 = "Hadrons"
    config.injection.lepton_injector.simulation.minimal_energy = 1e3
    config.injection.lepton_injector.simulation.maximal_energy = 1e4

    # Use the demo ice geo shipped in resources/
    config.detector.geo_file = "resources/geofiles/demo_ice.geo"

    # Force PPC as the photon propagator and allow re-use of a stale tmp dir
    config.photon_propagator.name = "PPC"
    config.photon_propagator.ppc.paths.force = True

    print("Initializing Prometheus (ice / PPC)")
    prom = Prometheus()
    print("Prometheus initialized")

    try:
        prom.sim()
    except Exception as e:
        print("Simulation error:", e)
        traceback.print_exc()
        sys.exit(1)

    print("Simulation completed successfully")


if __name__ == "__main__":
    main()
