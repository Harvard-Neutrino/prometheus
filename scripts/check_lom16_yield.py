#!/usr/bin/env python
"""Check that the shipped LOM-16 gives the published light yield when run in PPC.

Closure, in nearly clear ice: modules are placed on a cone around a cascade, so
that each receives the same light from a different direction. Dividing by a
vanilla DOM with a constant as.dat, whose effective area is known and the same in
all directions, gives the effective area of the LOM-16 against zenith and
wavelength, and the share of each PMT. This uses the curves the LOM-16 was made
from, so it tests PPC and Prometheus, not the module.

Distance, in real ice: LOM-16 photons over those of an IceCube DOM against Fig. 76 of
the Gen2 TDR. The light at a module is not isotropic in real ice, so the DOM is
averaged over six orientations. The ice model behind Fig. 76 is not known, and the
figure is against photon path length, which is longer than the distance, so this
comparison is approximate. Change the depth with --depth.
"""

import argparse
import importlib.util
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from fit_om_conf import pmt_response, zenith_profile

from prometheus.detector import Detector, Medium, Module, read_om_conf
from prometheus.particle import PropagatableParticle
from prometheus.photon_propagation.ppc_photon_propagator import ppc_sim

ROOT = Path(__file__).resolve().parent.parent
PPC_DIR = ROOT / "resources" / "PPC_executables" / "icecube-ppc"
LOM16 = ROOT / "resources" / "optical_modules" / "lom16"
DATA = ROOT / "examples" / "data" / "lom"
CENTER = np.array([0.0, 0.0, -2250.0])
CONE = np.radians(41.0)  # close to the peak of the cascade emission
DOM_TYPE = 1
# Orientations of the DOM in the distance check, as zenith and azimuth [deg].
FACING = [(180.0, 0.0), (0.0, 0.0), (90.0, 0.0), (90.0, 90.0), (90.0, 180.0), (90.0, 270.0)]

_spec = importlib.util.spec_from_file_location("gen1", ROOT / "examples/09_gen1_dom_comparison.py")
gen1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen1)


def unit_vector(zenith, azimuth):
    """Return unit vectors for angles in radians."""
    return np.stack(
        [np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)],
        axis=-1,
    )


def clear_ice(source, out):
    """Write ice tables with a uniform 1 km absorption and 100 km scattering length."""
    out.mkdir()
    for name in ("wv.dat", "rnd.txt"):
        shutil.copy(source / name, out)
    (out / "as.dat").write_text("1\n1\n")
    (out / "cfg.txt").write_text("1\n1.0\n0.35\n0.9\n")
    # A and B, the absorption that grows with wavelength, are turned off.
    (out / "icemodel.par").write_text("0 0\n0 0\n1e-9 0\n1 0\n")
    rows = [f"{1100 + 10 * i} 1e-6 1e-3 0" for i in range(170)]
    (out / "icemodel.dat").write_text("\n".join(rows) + "\n")


def real_ice(source, out, oversize):
    """Copy ice tables without IceCube's per-DOM files."""
    skip = shutil.ignore_patterns("geo-f2k", "eff-f2k", "dx.dat", "wv.rde")
    shutil.copytree(source, out, ignore=skip)
    lines = (out / "cfg.txt").read_text().splitlines()
    first = next(i for i, s in enumerate(lines) if s.strip() and not s.startswith("#"))
    lines[first] = str(oversize)
    lines[first + 1] = "1.0"
    (out / "cfg.txt").write_text("\n".join(lines) + "\n")


def simulate(positions, shape, axis, energy, tables, args, center=CENTER):
    """Return the hits of a cascade at ``center``, with the module index of each."""
    modules = [Module(center + p, (i + 1, 1), **shape) for i, p in enumerate(positions)]
    detector = Detector(modules, Medium.ICE)
    # A neutral pion is an electromagnetic cascade. PPC varies the light of hadronic
    # cascades from run to run, which would not cancel in the ratios.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "ppc"
        shutil.copytree(tables, work)
        particle = PropagatableParticle(111, energy, center, np.asarray(axis), 0, None, 0.0)
        config = {
            "paths": {
                "ppc_tmpdir": str(work),
                "ppc_tmpfile": "hits",
                "f2k_tmpfile": "losses",
                "ppctables": str(work),
                "ppc_exe": str(args.ppc),
                "om_tables": str(LOM16),
            },
            "simulation": {
                "device": args.device,
                "supress_output": True,
                "ppc_version": "official",
            },
        }
        ppc_sim(particle, detector, None, config)
    index = np.array([h.string_id - 1 for h in particle.hits], dtype=int)
    wavelength = np.array([h.wavelength for h in particle.hits])
    pmt = np.array([-1 if h.pmt_id is None else h.pmt_id for h in particle.hits], dtype=int)
    zenith = np.array([h.photon_zenith for h in particle.hits])
    azimuth = np.array([h.photon_azimuth for h in particle.hits])
    toward = -unit_vector(zenith, azimuth)
    return index, wavelength, pmt, toward


def cone_positions(axis, distance, n):
    """Return points at ``distance`` and the angle CONE from ``axis``."""
    helper = [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    phi = 2 * np.pi * (np.arange(n) + 0.37) / n
    around = np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v)
    return distance * (np.cos(CONE) * np.asarray(axis) + np.sin(CONE) * around)


def closure(args, lom):
    """Run the clear-ice check; return the figures of merit and plotted arrays."""
    axes = []
    for tilt in np.radians(args.tilts):
        axes.append([np.sin(tilt), 0.0, np.cos(tilt)])
    keys = ("cos", "hits", "reference", "expected", "wl_lom", "wl_ref", "counts", "shares")
    result = {key: [] for key in keys}
    pmt_axes = unit_vector(*np.radians(np.array(lom["pmt_dirs"])).T)
    sphere = sphere_points(20000)
    mean = pmt_response(sphere @ pmt_axes.T, lom["beta"]).sum(axis=1).mean()
    with tempfile.TemporaryDirectory() as tmp:
        tables = Path(tmp) / "clear"
        clear_ice(args.ice_tables, tables)
        for axis in axes:
            positions = cone_positions(axis, args.distance, args.n_modules)
            start = time.time()
            i, w, p, _ = simulate(positions, lom, axis, args.energy, tables, args)
            j, x, _, toward = simulate(positions, {}, axis, args.energy, tables, args)
            # The cascade is several meters long, so the photon directions are taken
            # from the reference, which accepts all directions equally.
            response = pmt_response(toward @ pmt_axes.T, lom["beta"])
            for k in range(len(positions)):
                here = j == k
                total = response[here].sum()
                result["cos"].append(toward[here, 2].mean())
                result["hits"].append((i == k).sum())
                result["reference"].append(here.sum())
                result["expected"].append(total / mean)
                result["counts"].append(np.bincount(p[i == k], minlength=lom["n_pmts"]))
                result["shares"].append(response[here].sum(axis=0) / total)
            result["wl_lom"].append(w)
            result["wl_ref"].append(x)
            n = np.bincount(i, minlength=len(positions))
            print(
                f"  axis zenith {np.degrees(np.arccos(axis[2])):5.1f}: LOM {n.sum()}, "
                f"reference {len(j)} photons, spread of reference over modules "
                f"{np.bincount(j, minlength=len(n)).std() / np.sqrt(len(j) / len(n)):.2f} sigma, "
                f"{time.time() - start:.0f} s",
                flush=True,
            )
    for key in ("wl_lom", "wl_ref"):
        result[key] = np.concatenate(result[key])
    return {key: np.array(value) for key, value in result.items()}


def reference_area(wavelength, ice_tables):
    """Return the area of the constant-as.dat DOM [cm^2]."""
    middle, area = gen1.effective_area(ice_tables)
    mean = gen1.sphere_mean(gen1.read_angular(ice_tables / "as.dat"))
    return np.interp(wavelength, middle, area / mean)


def report_closure(result, lom, args):
    """Print the closure comparison and return arrays for the figure."""
    curve_x, curve_y = np.loadtxt(DATA / "lom16_wavelength.txt", unpack=True)
    # Expected ratio of LOM to reference photons for a direction-averaged module. The
    # reference photons give the spectrum at the modules, which is not quite the
    # Cherenkov spectrum because the Cherenkov angle depends on wavelength.
    wavelengths = result["wl_ref"]
    weights = np.interp(wavelengths, curve_x, curve_y, left=0, right=0)
    wave_ratio = np.mean(weights / reference_area(wavelengths, args.ice_tables))
    edges = np.linspace(-1, 1, 11)
    tdr_x, tdr_y = np.loadtxt(DATA / "lom16_zenith_400nm.txt", unpack=True)
    scale = np.trapezoid(tdr_y, tdr_x) / 2
    print("  cos(zenith)   PPC     model    TDR    (area over its direction average)")
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (result["cos"] >= low) & (result["cos"] < high)
        if not inside.any():
            continue
        x = result["cos"][inside]
        n_lom = result["hits"][inside].sum()
        n_ref = result["reference"][inside].sum()
        value = n_lom / n_ref / wave_ratio
        error = value * np.sqrt(1 / n_lom + 1 / n_ref)
        expect = result["expected"][inside].sum() / n_ref
        published = np.mean(np.interp(x, tdr_x, tdr_y)) / scale
        rows.append((x.mean(), value, error, expect, published))
        print(f"  {x.mean():+.2f}        {value:.3f} +- {error:.3f}  {expect:.3f}  {published:.3f}")
    rows = np.array(rows)
    pull = (rows[:, 1] - rows[:, 3]) / rows[:, 2]
    print(
        f"  PPC / model: mean {np.mean(rows[:, 1] / rows[:, 3]):.4f}, "
        f"chi2 {np.sum(pull**2):.1f} for {len(pull)} bins"
    )
    print(f"  largest |PPC / TDR - 1|: {np.abs(rows[:, 1] / rows[:, 4] - 1).max():.3f}")

    # Wavelengths: correct the sampled directions to the sphere mean.
    factor = result["expected"].sum() / result["reference"].sum()
    bins = np.arange(290.0, 700.0, 20.0)
    n = np.histogram(result["wl_lom"], bins)[0]
    m = np.histogram(result["wl_ref"], bins)[0]
    center = (bins[1:] + bins[:-1]) / 2
    waves = []
    print("  wavelength   PPC area [cm^2]   input (TDR)")
    for k, (a, b) in enumerate(zip(n, m)):
        if a < 50 or b < 50:
            continue
        sub = np.arange(bins[k], bins[k + 1])
        ref = reference_area(sub, args.ice_tables) * gen1.cherenkov(sub)
        lom_expected = np.interp(sub, curve_x, curve_y) * gen1.cherenkov(sub)
        area = a / b * ref.sum() / gen1.cherenkov(sub).sum() / factor
        error = area * np.sqrt(1 / a + 1 / b)
        expect = lom_expected.sum() / gen1.cherenkov(sub).sum()
        waves.append((center[k], area, error, expect))
        print(f"  {center[k]:.0f}        {area:7.2f} +- {error:5.2f}   {expect:7.2f}")
    waves = np.array(waves)
    pull = (waves[:, 1] - waves[:, 3]) / waves[:, 2]
    print(f"  chi2 {np.sum(pull**2):.1f} for {len(pull)} bins")

    counts = result["counts"]
    total = counts.sum(axis=1, keepdims=True)
    expected = result["shares"] * total
    # Both the hits and the expected shares have statistical errors.
    variance = expected * (1 + total / result["reference"][:, None])
    use = expected > 10
    chi2 = np.sum((counts[use] - expected[use]) ** 2 / variance[use])
    pulls = np.abs(counts - expected)[use] / np.sqrt(variance[use])
    print(
        f"  PMT shares: chi2 {chi2:.0f} for {use.sum() - len(counts)} degrees of freedom; "
        f"largest deviation {pulls.max():.1f} sigma"
    )
    return rows, waves


def sphere_points(n):
    """Return roughly uniform unit vectors."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + np.sqrt(5)) * k
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def distance_check(args, lom):
    """Run the real-ice check."""
    x, y = np.loadtxt(DATA / "lom16_relative_yield.txt", unpack=True)
    center = np.array([0.0, 0.0, -args.depth])
    dom = {
        "module_type": DOM_TYPE,
        "Rr": gen1.RADIUS,
        "Rz": gen1.RADIUS,
        "beta": gen1.BETA,
        "n_pmts": 1,
    }
    cases = {
        "TDR DOM": np.loadtxt(DATA / "gen1_wavelength.txt"),
        "wv.dat DOM": np.column_stack(gen1.effective_area(args.ice_tables)),
    }
    axes = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    rows = []
    for distance, energy, oversize in args.distances:
        positions = distance * sphere_points(args.n_sphere)
        totals = {"LOM-16": 0, "TDR DOM": 0, "wv.dat DOM": 0}
        first = dict(totals)
        with tempfile.TemporaryDirectory() as tmp:
            for case in totals:
                tables = Path(tmp) / case
                real_ice(args.ice_tables, tables, oversize)
                shapes = [lom]
                if case != "LOM-16":
                    shapes = [{**dom, "pmt_dirs": [facing]} for facing in FACING]
                    np.savetxt(tables / f"om.wv_{DOM_TYPE}.0", cases[case], fmt="%.1f %.6f")
                for i, shape in enumerate(shapes):
                    for axis in axes:
                        index, _, _, _ = simulate(
                            positions, shape, np.array(axis, float), energy, tables, args, center
                        )
                        totals[case] += len(index)
                        if i == 0:
                            first[case] += len(index)
        ratios = []
        for case in ("TDR DOM", "wv.dat DOM"):
            value = totals["LOM-16"] / (totals[case] / len(FACING))
            ratios += [value, value * np.sqrt(1 / totals["LOM-16"] + 1 / totals[case])]
        rows.append((distance, *ratios, np.interp(distance, x, y)))
        print(
            f"  {distance:5.0f} m: photons {totals}; LOM-16 over TDR DOM {ratios[0]:.2f} +- "
            f"{ratios[1]:.2f}, over wv.dat DOM {ratios[2]:.2f} +- {ratios[3]:.2f}; "
            f"DOM facing down only {first['LOM-16'] / first['TDR DOM']:.2f}; "
            f"Fig. 76 {rows[-1][-1]:.2f}",
            flush=True,
        )
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ppc", type=Path, default=PPC_DIR / "gpu" / "ppc")
    parser.add_argument("--ice-tables", type=Path, default=PPC_DIR / "ice" / "spice_ftp-v3m")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--energy", type=float, default=3e4, help="clear-ice cascade [GeV]")
    parser.add_argument("--distance", type=float, default=30.0, help="clear-ice distance [m]")
    parser.add_argument("--n-modules", type=int, default=24, help="modules on the cone")
    parser.add_argument("--tilts", type=float, nargs="+", default=[0, 35, 70, 90, 110, 145, 180])
    parser.add_argument("--n-sphere", type=int, default=48)
    parser.add_argument(
        "--distances",
        type=lambda s: tuple(float(v) for v in s.split(":")),
        nargs="+",
        default=[(20, 2e3, 1), (50, 1e4, 1), (100, 5e4, 5), (150, 1e5, 5)],
        help="distance:energy:oversize",
    )
    parser.add_argument("--depth", type=float, default=2250.0, help="distance check [m]")
    parser.add_argument("--skip", choices=("closure", "distance"))
    parser.add_argument("--out", type=Path, help="figure")
    args = parser.parse_args()
    lom = read_om_conf(LOM16 / "om.conf")[171]
    plots = {}
    if args.skip != "closure":
        print("Closure in clear ice")
        plots["zenith"], plots["wave"] = report_closure(closure(args, lom), lom, args)
    if args.skip != "distance":
        print("Distance in real ice")
        plots["distance"] = distance_check(args, lom)
    if args.out:
        draw(plots, lom, args.out)


def draw(plots, lom, out):
    """Draw the comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    if "zenith" in plots:
        tdr_x, tdr_y = np.loadtxt(DATA / "lom16_zenith_400nm.txt", unpack=True)
        mean = np.trapezoid(tdr_y, tdr_x) / 2
        fine = np.linspace(-1, 1, 401)
        model = zenith_profile(fine, lom["pmt_dirs"], lom["beta"])
        rows = plots["zenith"]
        axes[0].plot(tdr_x, tdr_y, "k.", ms=4, label="TDR Fig. 75")
        axes[0].plot(fine, model / (np.trapezoid(model, fine) / 2) * mean, "C0-", label="fit")
        axes[0].errorbar(rows[:, 0], rows[:, 1] * mean, rows[:, 2] * mean, fmt="C3o", label="PPC")
        axes[0].set_xlabel(r"$\cos\theta$")
        axes[0].set_ylabel(r"Effective area at 400 nm [cm$^2$]")
        axes[0].set_ylim(100, 145)
        axes[0].legend()
        x, y = np.loadtxt(DATA / "lom16_wavelength.txt", unpack=True)
        axes[1].plot(x, y, "k.-", ms=4, label="TDR Fig. 75")
        waves = plots["wave"]
        axes[1].errorbar(waves[:, 0], waves[:, 1], waves[:, 2], fmt="C3o", label="PPC")
        axes[1].set_xlabel("Wavelength [nm]")
        axes[1].set_ylabel(r"Direction-averaged area [cm$^2$]")
        axes[1].legend()
    if "distance" in plots:
        x, y = np.loadtxt(DATA / "lom16_relative_yield.txt", unpack=True)
        rows = plots["distance"]
        axes[2].plot(x, y, "k-", label="TDR Fig. 76")
        axes[2].errorbar(rows[:, 0], rows[:, 1], rows[:, 2], fmt="C3o", label="PPC, TDR DOM")
        axes[2].errorbar(rows[:, 0], rows[:, 3], rows[:, 4], fmt="C1s", label="PPC, wv.dat DOM")
        axes[2].set_xlim(0, 200)
        axes[2].set_ylim(3, 4.5)
        axes[2].set_xlabel("Distance [m]")
        axes[2].set_ylabel("LOM-16 over IceCube DOM")
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
