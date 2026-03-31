#!/usr/bin/env python3
"""01_basic_water.py
Minimal water-case example to validate a Prometheus install.

Runs a single-event CPU-only simulation using the demo geo file.
"""
import sys
import traceback

try:
    from prometheus import Prometheus, config
except Exception as e:
    print("Error importing Prometheus:", e)
    print("Ensure you activated the venv and installed core requirements:")
    print("  source .venv/bin/activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Prefer CPU JAX when available
try:
    from jax.config import config as jconfig
    jconfig.update("jax_enable_x64", True)
    jconfig.update('jax_platform_name', 'cpu')
except Exception:
    pass


def main():
    # Minimal runtime configuration
    config['run']['run number'] = 1
    config['run']['random state seed'] = 1
    config['run']['nevents'] = 1

    # Injection: minimal LeptonInjector settings (may be skipped if missing)
    config['injection']['name'] = 'LeptonInjector'
    try:
        config['injection']['LeptonInjector']['simulation']['is ranged'] = False
        config['injection']['LeptonInjector']['simulation']['minimal energy'] = 1e3
        config['injection']['LeptonInjector']['simulation']['maximal energy'] = 1e4
    except Exception:
        # ensure keys exist in older/newer configs
        pass

    # Use the demo water geo shipped in resources (repo root path)
    config['detector']['geo file'] = 'resources/geofiles/demo_water.geo'

    print('Initializing Prometheus (minimal)')
    prom = Prometheus()
    print('Prometheus initialized')

    try:
        prom.sim()
    except Exception as e:
        print('Simulation error:', e)
        traceback.print_exc()
        sys.exit(1)

    print('Simulation completed successfully')


if __name__ == '__main__':
    main()
