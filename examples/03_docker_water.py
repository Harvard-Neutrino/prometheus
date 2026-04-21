#!/usr/bin/env python3
"""03_docker_water.py
Run the water-case Prometheus simulation inside a Docker container.

Uses the pre-built ``prometheus:cpu`` image by default; override with
``--image``.  The simulation parameters are forwarded into the container via
environment variables and an inline Python fragment, so no extra files need to
be mounted.

Output files (parquet + LeptonInjector HDF5) are written to ``--output-dir``
on the **host** by mounting that directory into the container at ``/output``.

Examples
--------
# Default (10 events, demo_water.geo, output in ./output):
python examples/03_docker_water.py

# Custom run:
python examples/03_docker_water.py \\
    --nevents 100 \\
    --geo resources/geofiles/orca.geo \\
    --seed 42 \\
    --min-energy 1e2 \\
    --max-energy 1e5 \\
    --ranged \\
    --output-dir /data/prometheus_runs
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import logging

logger = logging.getLogger(__name__)


# ── Available geo files (relative to repo root / mounted at /opt/prometheus) ──
GEO_CHOICES = [
    "resources/geofiles/demo_water.geo",
    "resources/geofiles/orca.geo",
    "resources/geofiles/arca.geo",
    "resources/geofiles/gvd.geo",
    "resources/geofiles/pone_triangle.geo",
    "resources/geofiles/trident.geo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Prometheus water simulation inside a Docker container.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=10,
        metavar="N",
        help="Number of events to simulate.",
    )
    parser.add_argument(
        "--geo",
        default="resources/geofiles/demo_water.geo",
        metavar="PATH",
        help=(
            "Geo file path relative to the repo root.  "
            "Choices: " + ", ".join(GEO_CHOICES)
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="SEED",
        help="Random state seed.",
    )
    parser.add_argument(
        "--ranged",
        action="store_true",
        default=False,
        help="Use ranged (volume) injection instead of the default unranged mode.",
    )
    parser.add_argument(
        "--min-energy",
        type=float,
        default=1e3,
        dest="min_energy",
        metavar="GeV",
        help="Minimal injection energy [GeV].",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        default=1e4,
        dest="max_energy",
        metavar="GeV",
        help="Maximal injection energy [GeV].",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        metavar="DIR",
        dest="output_dir",
        help="Host directory where output files are written (created if absent).",
    )
    parser.add_argument(
        "--image",
        default="prometheus:cpu",
        metavar="IMAGE",
        help="Docker image to use.",
    )
    return parser.parse_args()


# ── Python snippet executed inside the container ──────────────────────────────
# Uses only the standard prometheus API; no file I/O assumptions.
INNER_SCRIPT = textwrap.dedent(
    """\
    import os, sys

    try:
        from jax.config import config as jconfig
        jconfig.update("jax_enable_x64", True)
        jconfig.update("jax_platform_name", "cpu")
    except Exception:
        pass

    from prometheus import Prometheus, config

    config.run.run_number = 1
    config.run.random_state_seed = int(os.environ['PROM_SEED'])
    config.run.nevents = int(os.environ['PROM_NEVENTS'])
    config.run.storage_prefix = '/output/'

    config.injection.name = 'LeptonInjector'
    config.injection.lepton_injector.simulation.is_ranged = os.environ['PROM_RANGED'] == '1'
    config.injection.lepton_injector.simulation.minimal_energy = float(os.environ['PROM_MIN_ENERGY'])
    config.injection.lepton_injector.simulation.maximal_energy = float(os.environ['PROM_MAX_ENERGY'])

    config.detector.geo_file = os.environ['PROM_GEO']

    print('Initializing Prometheus')
    prom = Prometheus()
    print('Running simulation')
    prom.sim()
    print('Simulation completed successfully')
    """
)


def main() -> None:
    args = parse_args()

    if not shutil.which("docker"):
        logger.error("'docker' not found on PATH. Install Docker or add it to PATH.")
        logger.info("Hint: https://docs.docker.com/get-docker/")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--volume", f"{output_dir}:/output",
        "--env", f"PROM_NEVENTS={args.nevents}",
        "--env", f"PROM_GEO=/opt/prometheus/{args.geo}",
        "--env", f"PROM_SEED={args.seed}",
        "--env", f"PROM_RANGED={'1' if args.ranged else '0'}",
        "--env", f"PROM_MIN_ENERGY={args.min_energy}",
        "--env", f"PROM_MAX_ENERGY={args.max_energy}",
        args.image,
        "python", "-c", INNER_SCRIPT,
    ]

    print(f"Image:      {args.image}")
    print(f"Events:     {args.nevents}")
    print(f"Geo:        {args.geo}")
    print(f"Seed:       {args.seed}")
    print(f"Ranged:     {args.ranged}")
    print(f"Energy:     [{args.min_energy:.3g}, {args.max_energy:.3g}] GeV")
    print(f"Output dir: {output_dir}")
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
