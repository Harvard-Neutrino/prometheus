#!/usr/bin/env python3
"""07_numu_lom_array.py
Simulate muon neutrinos in a hexagonal array of LOM-16 strings.

The neutrinos come from a stored LeptonInjector file, so their energies, zenith
angles and inelasticities were sampled from the interaction model. By default one
event is moved to --vertex. With --n-events, the stored events below --max-energy
are repeated with random azimuths and vertices. Either way the events are
illustrations and cannot be weighted. PPC takes about 1 s per TeV on a CPU.

Writes the photons (parquet) and the module positions (geo).

Usage
-----
python examples/07_numu_lom_array.py
python examples/08_event_display.py examples/output/numu_lom/1_photons.parquet \\
    --geo examples/output/numu_lom/detector.geo
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from prometheus import Prometheus, config
from prometheus.detector import Detector, Medium, Module, read_om_conf

REPO_ROOT = Path(__file__).resolve().parent.parent
PPC_DIR = REPO_ROOT / "resources" / "PPC_executables" / "icecube-ppc"
LOM16_DIR = REPO_ROOT / "resources" / "optical_modules" / "lom16"
EVENTS = REPO_ROOT / "tests" / "resources" / "MuMinus_Hadrons_seed_925_LI_output.h5"
LOM16_TYPE = 171
VERTEX_MARGIN = 30.0  # m, around the detector, for random vertices


def hexagonal_strings(n_rings, spacing):
    """Return (x, y) of a hexagonal grid with rings around a central string."""
    positions = []
    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if abs(i + j) <= n_rings:
                positions.append((spacing * (i + j / 2), spacing * j * np.sqrt(3) / 2))
    return positions


def make_detector(args):
    """Build the array of LOM-16 modules."""
    lom16 = read_om_conf(LOM16_DIR / "om.conf")[LOM16_TYPE]
    z = args.z_center + args.module_spacing * (np.arange(args.n_modules) - (args.n_modules - 1) / 2)
    modules = []
    for string, (x, y) in enumerate(hexagonal_strings(args.rings, args.string_spacing)):
        for i, zi in enumerate(z):
            modules.append(Module(np.array([x, y, zi]), (string + 1, i + 1), **lom16))
    return Detector(modules, Medium.ICE)


def write_events(source_path, out_path, indices, vertices, rotations):
    """Copy LeptonInjector events, moved to ``vertices`` and rotated in azimuth [rad]."""
    with h5py.File(source_path) as source, h5py.File(out_path, "w") as out:
        name = next(iter(source))
        for key, dataset in source[name].items():
            rows = dataset[:][indices]
            if key == "properties":
                for column, axis in enumerate("xyz"):
                    rows[axis] = vertices[:, column]
                rows["azimuth"] = (rows["azimuth"] + rotations) % (2 * np.pi)
            else:
                rows["Position"] = vertices
                rows["Direction"][:, 1] += rotations
            out.create_dataset(f"{name}/{key}", data=rows)


def choose_events(args, detector):
    """Return the event indices, vertices and azimuth rotations to simulate."""
    if args.n_events is None:
        return np.array([args.event]), np.array([args.vertex], dtype=float), np.zeros(1)
    with h5py.File(EVENTS) as source:
        energies = source[next(iter(source))]["properties"]["totalEnergy"]
    usable = np.flatnonzero(energies < args.max_energy)
    if len(usable) == 0:
        raise SystemExit(f"No stored event is below {args.max_energy:g} GeV")
    rng = np.random.default_rng(args.seed)
    n = args.n_events
    low = detector.module_coords.min(axis=0)
    high = detector.module_coords.max(axis=0)
    center = (low + high) / 2
    radius = (np.hypot(*(detector.module_coords - center)[:, :2].T).max() + VERTEX_MARGIN) * (
        np.sqrt(rng.uniform(size=n))
    )
    angle = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(low[2] - VERTEX_MARGIN, high[2] + VERTEX_MARGIN, n)
    vertices = np.column_stack(
        [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), z]
    )
    return np.resize(usable, n), vertices, rng.uniform(0, 2 * np.pi, n)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--event", type=int, default=12, help="event in the stored file")
    parser.add_argument(
        "--vertex", type=float, nargs=3, default=(-135.0, 50.0, -2410.0), metavar=("X", "Y", "Z")
    )
    parser.add_argument("--n-events", type=int, help="simulate this many random events instead")
    parser.add_argument("--max-energy", type=float, default=2e4, help="[GeV], with --n-events")
    parser.add_argument("--rings", type=int, default=2, help="hexagonal rings of strings")
    parser.add_argument("--string-spacing", type=float, default=60.0)
    parser.add_argument("--module-spacing", type=float, default=20.0)
    parser.add_argument("--n-modules", type=int, default=20)
    parser.add_argument("--z-center", type=float, default=-2290.0)
    parser.add_argument("--ppc", type=Path, default=PPC_DIR / "gpu" / "ppc")
    parser.add_argument("--ice-tables", type=Path, default=PPC_DIR / "ice" / "spice_ftp-v3m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "examples" / "output" / "numu_lom"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = make_detector(args)
    detector.to_geo(str(args.out_dir / "detector.geo"))

    with tempfile.TemporaryDirectory() as tmp:
        events = Path(tmp) / "event.h5"
        indices, vertices, rotations = choose_events(args, detector)
        write_events(EVENTS, events, indices, vertices, rotations)
        # IceCube's per-DOM files would be applied to modules with the same key.
        tables = Path(tmp) / "ice"
        skip = shutil.ignore_patterns("geo-f2k", "eff-f2k", "dx.dat")
        shutil.copytree(args.ice_tables, tables, ignore=skip)

        config.run.run_number = 1
        config.run.nevents = len(indices)
        config.run.random_state_seed = args.seed
        config.run.storage_prefix = f"{args.out_dir}/"
        config.injection.name = "LeptonInjector"
        config.injection.lepton_injector.inject = False
        config.injection.lepton_injector.paths.injection_file = str(events)
        config.photon_propagator.name = "PPC"
        ppc = config.photon_propagator.ppc
        ppc.paths.ppc_exe = str(args.ppc)
        ppc.paths.ppctables = str(tables)
        ppc.paths.om_tables = str(LOM16_DIR)
        ppc.simulation.ppc_version = "official"
        ppc.simulation.output_mode = "extended"
        Prometheus(config, detector=detector).sim()


if __name__ == "__main__":
    main()
