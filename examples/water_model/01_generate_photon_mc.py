#!/usr/bin/env python3
"""01_generate_photon_mc.py
Generate KM3NeT Olympus training data using photon Monte Carlo.

Thin CLI wrapper around
:func:`hyperion.models.photon_arrival_time_nflow.training_data.generate_training_data` —
see that module's docstring for the physics and the training-data schema.

Output
------
  <out_dir>/shape_data.npz   — (log10_dist, angle, t_residual, weight)
  <out_dir>/counts_data.npz  — (log10_dist, angle, log10_survival, n_detected)

Usage
-----
    .prometheus_env/bin/python examples/water_model/01_generate_photon_mc.py

    # Custom settings
    .prometheus_env/bin/python examples/water_model/01_generate_photon_mc.py \\
        --n-photons 200000 --n-distances 60 --d-max 200 --out-dir examples/output/water_model
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from hyperion.models.photon_arrival_time_nflow.training_data import (
        DOM_RADIUS,
        generate_training_data,
    )
except Exception:
    logger.exception("hyperion/JAX not available — run from the Prometheus repository root.")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate KM3NeT Olympus photon-MC training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-photons",
        type=int,
        default=100_000,
        help="Photons emitted per source distance.",
    )
    p.add_argument(
        "--n-distances",
        type=int,
        default=50,
        help="Number of source distances (log-uniform from d-min to d-max).",
    )
    p.add_argument(
        "--n-angle-bins",
        type=int,
        default=20,
        help="Emission-angle bins for counts data.",
    )
    p.add_argument(
        "--d-min",
        type=float,
        default=DOM_RADIUS + 0.01,
        help="Min source distance [m]. Must be > DOM radius (0.30 m) to avoid the "
        "source sitting exactly on the sphere surface, which causes d=0 "
        "intersection tests to fail the strict d>0 check and suppresses hits.",
    )
    p.add_argument("--d-max", type=float, default=150.0, help="Max source distance [m].")
    p.add_argument("--wl-min", type=float, default=290.0, help="Min Cherenkov wavelength [nm].")
    p.add_argument("--wl-max", type=float, default=700.0, help="Max Cherenkov wavelength [nm].")
    p.add_argument(
        "--max-time",
        type=float,
        default=3000.0,
        help="Max photon propagation time [ns] (cutoff for un-intersected photons).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "examples" / "output" / "water_model"),
        metavar="DIR",
        help="Output directory for NPZ training data.",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_data, counts_data = generate_training_data(
        n_photons=args.n_photons,
        n_distances=args.n_distances,
        n_angle_bins=args.n_angle_bins,
        d_min=args.d_min,
        d_max=args.d_max,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        max_time=args.max_time,
        seed=args.seed,
    )

    if shape_data is not None:
        shape_npz = out_dir / "shape_data.npz"
        np.savez(shape_npz, **shape_data)
        print(f"\nShape data : {len(shape_data['log10_dist']):,} samples (weighted) → {shape_npz}")
    else:
        print("\nShape data : no detected photons — increase --n-photons")

    if counts_data is not None:
        counts_npz = out_dir / "counts_data.npz"
        np.savez(counts_npz, **counts_data)
        print(f"Counts data: {len(counts_data['log10_dist']):,} samples → {counts_npz}")
    else:
        print("Counts data: no bins populated — increase --n-photons")

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
