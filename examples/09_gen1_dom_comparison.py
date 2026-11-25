#!/usr/bin/env python3
"""09_gen1_dom_comparison.py
Compare an IceCube-like DOM in PPC's multi-PMT mode with the vanilla PPC DOM.

The multi-PMT module has one PMT pointing down, with the shape given in the PPC
documentation (beta = 0.49, radius 0.1651 m). Its wavelength table is computed
from the wv.dat and as.dat that vanilla PPC uses, so that the direction-averaged
effective areas agree. Three cases are run on the same cascades:

    vanilla           legacy DOM with the as.dat of the ice tables
    vanilla, f_beta   legacy DOM with as.dat replaced by a polynomial fit to f_beta
    multi-PMT         the om.conf module

The last two have the same angular response up to the polynomial fit and should
agree. The first differs from them in angular response, because as.dat describes
a DOM in hole ice and f_beta a bare one.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from prometheus.detector import Detector, Medium, Module
from prometheus.particle import PropagatableParticle
from prometheus.photon_propagation.ppc_photon_propagator import ppc_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fit_om_conf import pmt_response  # noqa: E402

PPC_DIR = REPO_ROOT / "resources" / "PPC_executables" / "icecube-ppc"
RADIUS = 0.16510  # m, OMR in PPC
BETA = 0.49
TYPE_ID = 1
PHOTONS_PER_METER = 2450.08  # dppm in PPC
AS_DEGREE = 10
CENTER = np.array([0.0, 0.0, -2250.0])
STRINGS = [(30.0, 0.0), (-30.0, 0.0), (0.0, 30.0), (0.0, -30.0)]


def cherenkov(wavelength):
    """Return PPC's Cherenkov spectrum [0.1 / m / nm] (``spec::cherenkov``)."""
    w = wavelength * 1e-3
    n = 1.55749 - w * (1.57988 - w * (3.99993 - w * (4.68271 - 2.09354 * w)))
    return np.pi * 0.2 / 137.03599911 * (1 - 1 / n**2) / w**2


def read_angular(path):
    """Return the polynomial coefficients of an as.dat file, lowest first."""
    values = np.loadtxt(path).ravel()
    if values[0] <= 0:
        raise SystemExit(f"{path} uses the surface-sensitivity format, which has no polynomial")
    return values[1:]


def sphere_mean(coefficients):
    """Return the mean over the sphere of a polynomial in cos(theta)."""
    return sum(c / (i + 1) for i, c in enumerate(coefficients) if i % 2 == 0)


def effective_area(tables):
    """Return wavelengths [nm] and the direction-averaged effective area [cm^2]."""
    quantile, wavelength = np.loadtxt(tables / "wv.dat", unpack=True)
    middle = (wavelength[1:] + wavelength[:-1]) / 2
    slope = np.diff(quantile) / np.diff(wavelength)
    mean = sphere_mean(read_angular(tables / "as.dat"))
    area = np.pi * RADIUS**2 * PHOTONS_PER_METER * slope * mean / cherenkov(middle)
    return middle, area


def write_lab_angular(path, average):
    """Write an as.dat with the shape of f_beta and the given sphere mean."""
    x = np.linspace(-1, 1, 2001)
    coefficients = np.polynomial.polynomial.polyfit(x, pmt_response(x, BETA), AS_DEGREE)
    coefficients *= average / sphere_mean(coefficients)
    curve = np.polynomial.polynomial.polyval(x, coefficients)
    with open(path, "w") as f:
        f.write("\n".join(f"{v:.8f}" for v in [curve.max(), *coefficients]) + "\n")
    target = pmt_response(x, BETA) * average / pmt_response(x, BETA).mean()
    return np.abs(curve - target).max()


def make_detector(case, spacing, n_modules):
    """Return four strings of DOMs around the cascade."""
    shape = {}
    if case == "multi-PMT":
        shape = {
            "module_type": TYPE_ID,
            "Rr": RADIUS,
            "Rz": RADIUS,
            "beta": BETA,
            "n_pmts": 1,
            "pmt_dirs": [(180.0, 0.0)],
        }
    z = CENTER[2] + spacing * (np.arange(n_modules) - (n_modules - 1) / 2)
    modules = []
    for string, (x, y) in enumerate(STRINGS):
        for i, zi in enumerate(z):
            modules.append(Module(np.array([x, y, zi]), (string + 1, i + 1), **shape))
    return Detector(modules, Medium.ICE)


def run(case, sign, args):
    """Return the height of the module [m] and the time [ns] of every photon."""
    detector = make_detector(case, args.spacing, args.n_modules)
    with tempfile.TemporaryDirectory() as tmp:
        tables = Path(tmp) / "ppc"
        skip = shutil.ignore_patterns("geo-f2k", "eff-f2k", "dx.dat", "om.*")
        shutil.copytree(args.ice_tables, tables, ignore=skip)
        if case == "multi-PMT":
            np.savetxt(
                tables / f"om.wv_{TYPE_ID}.0",
                np.column_stack(effective_area(args.ice_tables)),
                fmt="%.1f %.6f",
            )
        elif case == "vanilla, f_beta":
            mean = sphere_mean(read_angular(args.ice_tables / "as.dat"))
            write_lab_angular(tables / "as.dat", mean)
        if args.oversize is not None:
            lines = (tables / "cfg.txt").read_text().splitlines()
            first = next(i for i, s in enumerate(lines) if s.strip() and not s.startswith("#"))
            lines[first] = str(args.oversize)
            (tables / "cfg.txt").write_text("\n".join(lines) + "\n")
        # A neutral pion is an electromagnetic cascade. PPC varies the light of hadronic
        # cascades from run to run, which would not cancel in the comparison.
        particle = PropagatableParticle(
            111, args.energy, CENTER, np.array([0.0, 0.0, sign]), 0, None, 0.0
        )
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
    z = np.array([detector[(h.string_id, h.om_id)].pos[2] for h in particle.hits]) - CENTER[2]
    return z, np.array([h.time for h in particle.hits])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ppc", type=Path, default=PPC_DIR / "gpu" / "ppc")
    parser.add_argument("--ice-tables", type=Path, default=PPC_DIR / "ice" / "spice_ftp-v3m")
    parser.add_argument("--device", type=int, default=0, help="GPU, or random sequence on CPU")
    parser.add_argument("--energy", type=float, default=3e4, help="cascade energy [GeV]")
    parser.add_argument("--spacing", type=float, default=17.0)
    parser.add_argument("--n-modules", type=int, default=20)
    parser.add_argument("--oversize", type=float, help="DOM oversize factor (default: cfg.txt)")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "examples" / "output" / "gen1_dom.png"
    )
    args = parser.parse_args()

    wavelength, area = effective_area(args.ice_tables)
    print(
        f"Direction-averaged effective area at 400 nm: {np.interp(400, wavelength, area):.1f} cm^2"
    )
    with tempfile.TemporaryDirectory() as tmp:
        worst = write_lab_angular(Path(tmp) / "as.dat", 0.34)
    print(f"Largest error of the polynomial for f_beta (peak near 1): {worst:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ("vanilla", "vanilla, f_beta", "multi-PMT")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    edges = args.spacing * (np.arange(args.n_modules + 1) - args.n_modules / 2)
    middle = (edges[1:] + edges[:-1]) / 2
    for column, sign in enumerate((1.0, -1.0)):
        for case, style in zip(cases, ("C1-", "C2--", "C0-")):
            z, t = run(case, sign, args)
            below = int((z < 0).sum())
            print(
                f"{'up' if sign > 0 else 'down'}going  {case:16s} {len(z):7d} hits, "
                f"above {len(z) - below}, below {below}, median time {np.median(t):.1f} ns"
            )
            counts = np.histogram(z, edges)[0]
            axes[0, column].plot(middle, counts, style, label=case)
            times = np.histogram(t, np.linspace(100, 1500, 57))[0]
            axes[1, column].stairs(times, np.linspace(100, 1500, 57), color=style[:2], ls=style[2:])
        name = "Upgoing" if sign > 0 else "Downgoing"
        axes[0, column].set_title(f"{name} {args.energy / 1e3:g} TeV cascade")
        axes[0, column].set_xlabel("Module height above cascade [m]")
        axes[1, column].set_xlabel("Photon time [ns]")
        for axis in axes[:, column]:
            axis.set_yscale("log")
            axis.set_ylabel("Photons")
    axes[0, 0].legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
