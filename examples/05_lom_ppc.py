#!/usr/bin/env python3
"""05_lom_ppc.py
Simulate IceCube-Gen2 LOMs with the official IceCube PPC.

Reads module definitions from a directory with om.conf and om.wv_* files, builds
one string of modules, places a hadronic cascade beside it, and prints the hits
per PMT. The default directory is the LOM-16 shipped with Prometheus. Other
directories and type IDs can be given for comparison; see examples/README.md.

With --out, the photons are written in Prometheus's parquet format, which
04_event_view.py can draw.
"""

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np

from prometheus.detector import Detector, Medium, Module, read_om_conf
from prometheus.particle import PropagatableParticle
from prometheus.photon_propagation.ppc_photon_propagator import ppc_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
PPC_DIR = REPO_ROOT / "resources" / "PPC_executables" / "icecube-ppc"

LOM16_DIR = REPO_ROOT / "resources" / "optical_modules" / "lom16"


def make_tables(ice_tables, om_dir, type_id, out_dir):
    """Combine the ice tables with the module tables PPC needs for one type."""
    wavelength_tables = sorted(om_dir.glob(f"om.wv_{type_id}.*"))
    if not wavelength_tables:
        raise FileNotFoundError(f"No om.wv_{type_id}.* in {om_dir}")
    # Skip IceCube's per-DOM files: they are keyed by string and module number and
    # would be applied to the unrelated modules simulated here.
    skip = shutil.ignore_patterns("geo-f2k", "eff-f2k", "dx.dat", "om.*")
    shutil.copytree(ice_tables, out_dir, ignore=skip)
    for table in wavelength_tables:
        shutil.copy(table, out_dir)


def simulate(args, type_id):
    """Return the result of one cascade on a string of modules of one type."""
    types = read_om_conf(args.om_dir / "om.conf")
    if type_id not in types:
        raise SystemExit(f"Type {type_id} is not in {args.om_dir}; available: {sorted(types)}")
    module_kwargs = types[type_id]
    z_center = -2150.0
    z = z_center + args.spacing * (np.arange(args.n_modules) - (args.n_modules - 1) / 2)
    modules = [
        Module(np.array([0.0, 0.0, zi]), (1, i + 1), **module_kwargs) for i, zi in enumerate(z)
    ]
    detector = Detector(modules, Medium.ICE)

    # A charged pion is treated as a point-like hadronic cascade.
    particle = PropagatableParticle(
        211,
        args.energy,
        np.array([args.distance, 0.0, z_center]),
        np.array([0.0, 0.0, 1.0]),
        0,
        None,
        0.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tables = Path(tmp) / "ppc"
        make_tables(args.ice_tables, args.om_dir, type_id, tables)
        config = {
            "paths": {
                "ppc_tmpdir": str(tables),
                "ppc_tmpfile": "hits",
                "f2k_tmpfile": "losses",
                "ppctables": str(tables),
                "ppc_exe": str(args.ppc),
            },
            "simulation": {
                "device": args.device,
                "supress_output": True,
                "ppc_version": "official",
            },
        }
        ppc_sim(particle, detector, None, config)
    return module_kwargs, detector, particle


def write_parquet(path, detector, particle):
    """Write the hits and the cascade with the field names Prometheus uses."""
    hits = particle.hits
    positions = np.array([detector[(h.string_id, h.om_id)].pos for h in hits])
    photons = {
        "sensor_pos_x": positions[:, 0],
        "sensor_pos_y": positions[:, 1],
        "sensor_pos_z": positions[:, 2],
        "string_id": [h.string_id for h in hits],
        "sensor_id": [h.om_id for h in hits],
        "t": [h.time for h in hits],
    }
    for name in (
        "wavelength",
        "photon_zenith",
        "photon_azimuth",
        "om_zenith",
        "om_azimuth",
        "pmt_id",
    ):
        photons[name] = [getattr(h, name) for h in hits]
    mc_truth = {
        "initial_state_energy": float(particle.e),
        "initial_state_type": int(particle.pdg_code),
        "initial_state_x": float(particle.position[0]),
        "initial_state_y": float(particle.position[1]),
        "initial_state_z": float(particle.position[2]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(ak.Array([{"mc_truth": mc_truth, "photons": photons}]), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--om-dir", type=Path, default=LOM16_DIR, help="directory with om.conf and om.wv_*"
    )
    parser.add_argument("--types", type=int, nargs="+", default=[171], help="type IDs in om.conf")
    parser.add_argument("--ppc", type=Path, default=PPC_DIR / "gpu" / "ppc")
    parser.add_argument("--ice-tables", type=Path, default=PPC_DIR / "ice" / "spice_ftp-v3m")
    parser.add_argument("--device", type=int, default=0, help="GPU, or random sequence on CPU")
    parser.add_argument("--energy", type=float, default=1e4, help="cascade energy [GeV]")
    parser.add_argument("--distance", type=float, default=30.0, help="distance to string [m]")
    parser.add_argument("--spacing", type=float, default=17.0, help="module spacing [m]")
    parser.add_argument("--n-modules", type=int, default=20)
    parser.add_argument("--out", type=Path, help="photon parquet file (one module type only)")
    args = parser.parse_args()
    if args.out and len(args.types) != 1:
        parser.error("--out needs exactly one entry in --types")

    for type_id in args.types:
        module_kwargs, detector, particle = simulate(args, type_id)
        hits = particle.hits
        counts = Counter(h.pmt_id for h in hits)
        print(f"Type {type_id}: {len(hits)} hits")
        for pmt, (zenith, azimuth) in enumerate(module_kwargs["pmt_dirs"]):
            print(f"  PMT {pmt:2d}  zenith {zenith:5.1f}  azimuth {azimuth:6.1f}  {counts[pmt]}")
        if args.out:
            if not hits:
                parser.error(f"no hits to write to {args.out}")
            write_parquet(args.out, detector, particle)
            print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
