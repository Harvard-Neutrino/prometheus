#!/usr/bin/env python3
"""06_fit_lom_shape.py
Fit the PPC description of the IceCube-Gen2 LOM-16 to its published response.

PPC does not simulate the inside of a module. It describes each PMT by one
function, a flat disc topped by a spherical cap,

    f(x; beta),   x = cosine between the PMT axis and the direction to the light,

and the module by the sum F over its PMTs. beta = 1 is a flat disc and beta = 0
a hemisphere. ``pmt_response`` in scripts/fit_om_conf.py is that function.

What is fitted
    The effective area of the LOM-16 against zenith angle at 400 nm, A_i at
    cos(zenith)_i. The curve was simulated with Geant4 by IceCube-Gen2, so this
    reproduces another simulation, not a measurement. The published figure is
    taken to be averaged over azimuth, as the model is.

Free parameters
    The zenith angle of the four PMTs nearest each pole, the zenith angle of the
    four next to them, and beta. The lower half mirrors the upper half.

Minimized quantity
    sum_i [(s F(zenith_i) - A_i) / mean(A)]^2
    The scale s takes its best value at every step, so only the shape is used.
    The size and wavelength dependence come from the om.wv table.

The fitted angles are effective. They differ from the mechanical PMT tilts
because they also absorb the glass, gel pads and shadowing that PPC leaves out.
An azimuth-averaged curve cannot say which ring has which angle, so the order of
the starting point is kept, and beta between 0.9 and 1 fits about equally well.

Inputs and citations are in examples/data/lom/README.md. No input from a private
repository is used.

Outputs, in --out-dir
    om.conf         module definition with the fitted values
    om.wv_171.0     wavelength [nm] and direction-averaged effective area [cm^2]
    lom16_fit.png   curve, fit, and the model with mechanical tilts

PPC also needs om.dirs, which is not produced here.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fit_om_conf import best_scale, fit, format_module, shape_rms, zenith_profile  # noqa: E402

from prometheus.detector import read_om_conf  # noqa: E402

DATA = REPO_ROOT / "examples" / "data" / "lom"
TYPE_ID = 171


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "examples" / "output" / "lom16_fit"
    )
    args = parser.parse_args()

    # 1. Published effective area, and the mechanical layout as a starting point.
    cos_zenith, area = np.loadtxt(DATA / "lom16_zenith_400nm.txt", unpack=True)
    module = read_om_conf(DATA / "lom16_start.conf")[TYPE_ID]

    # 2. Fit the two ring angles and beta.
    result = fit(cos_zenith, area, module)
    before = shape_rms(cos_zenith, area, module["pmt_dirs"], module["beta"])
    rings = ", ".join(f"{angle:.1f}" for angle in result.ring_angles)
    print(f"Mechanical tilts: RMS residual {100 * before:.1f}% of the mean area")
    print(f"Fit:              RMS residual {100 * result.rms:.1f}% of the mean area")
    print(f"  beta = {result.beta:.3f}, rings = {rings} deg")
    if not result.success or result.at_bound:
        print(f"  WARNING: converged={result.success}, at a limit: {result.at_bound}")

    # 3. Write the PPC files.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conf = args.out_dir / "om.conf"
    conf.write_text(
        "# name module area beta Rr Rz num dir cable\n"
        + format_module("LOM-16", module, result)
        + "\n"
    )
    table = np.loadtxt(DATA / "lom16_wavelength.txt")
    # PPC cannot skip comment lines in this file.
    np.savetxt(args.out_dir / f"om.wv_{TYPE_ID}.0", table, fmt="%.0f %.3f")

    # 4. Compare.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(cos_zenith, area, "k.", label="TDR Fig. 75 (Geant4)")
    curves = (
        ("fit", result.pmt_dirs, result.beta, "C0-"),
        ("mechanical tilts", module["pmt_dirs"], module["beta"], "C1--"),
    )
    for label, dirs, beta, style in curves:
        model = zenith_profile(cos_zenith, dirs, beta)
        ax.plot(cos_zenith, best_scale(model, area) * model, style, label=label)
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel(r"Effective area at 400 nm [cm$^2$]")
    ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "lom16_fit.png", dpi=150)
    print(f"Saved {args.out_dir}")


if __name__ == "__main__":
    main()
