#!/usr/bin/env python3
"""Physics checks for an Olympus counts model, before it is used or shipped.

The counts network maps ``[log10(distance_m), viewing_angle_rad]`` to
``log10`` of the fraction of a source's photons that reach a module, and the
propagator multiplies that fraction by every photon the source emits
(``norm_flow_photons.py``). Four things must therefore hold for any model,
whatever water it describes:

1. **Geometric bound.** A module of radius ``R`` subtends ``pi R^2 / d^2``
   steradians. Even a source that beams every photon into one steradian
   cannot put more than ``pi R^2 / d^2`` of them onto the module, so the
   fraction at any angle must stay below ``0.283 / d^2`` (``R`` = 0.30 m).
   The ANTARES low-E model shipped in 2026-07 broke this by 10^3-10^4.
2. **Cherenkov-cone convention.** The angle is the one between the SOURCE
   direction and the source-to-module vector, so the fraction must peak
   near the Cherenkov angle (42 deg in water), not at 0 or 180 deg. A model
   trained on isotropic emission binned by the photon's own emission angle
   peaks at 0 deg and is the wrong quantity for the propagator.
3. **Attenuation.** Beyond a few module radii the angle-averaged
   ``d^2 x fraction`` must fall monotonically, with an effective length of
   the order of the medium's absorption length (KM3NeT: 27-65 m over the
   Cherenkov band). A network extrapolating beyond its data typically goes
   flat or rises here.
4. **Training domain.** Every module distance the detector will ask for
   must lie inside the range the model was trained on; outside it the
   network is guessing. The range is read from the pickle's config when the
   training recorded it (``train_d_min`` / ``train_d_max``), and reported
   as unknown otherwise.

With ``--counts-data`` the model is also compared with the rows it was
trained on (RMS in dex over bins with >= 2 detected photons).

    .prometheus_env/bin/python examples/water_model/03_check_photon_model.py \\
        --counts resources/olympus_resources/photon_arrival_time_counts_params.pickle
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DOM_RADIUS_M = 0.30
CHERENKOV_DEG = 42.0
# KM3NeT absorption lengths [m] vs wavelength [nm] (hyperion training_data.py).
_ABS_WL = np.array([350.0, 380.0, 400.0, 420.0, 450.0, 470.0, 500.0, 520.0, 550.0, 600.0])
_ABS_LEN = np.array([8.0, 14.0, 22.0, 34.0, 52.0, 62.0, 65.0, 60.0, 45.0, 20.0])


def spectrum_weighted_absorption(wl_min: float = 290.0, wl_max: float = 700.0) -> float:
    """Cherenkov-spectrum (1/lambda^2) weighted absorption length [m]."""
    wl = np.linspace(max(wl_min, _ABS_WL[0]), min(wl_max, _ABS_WL[-1]), 200)
    w = 1.0 / wl**2
    return float(np.sum(w * np.interp(wl, _ABS_WL, _ABS_LEN)) / np.sum(w))


def load_counts(path: Path):
    import jax.numpy as jnp

    from hyperion.models.photon_arrival_time_nflow.net import make_counts_net_fn
    from prometheus.photon_propagation.olympus.event_generation.photon_propagation import (
        norm_flow_photons,
    )

    haiku_load = norm_flow_photons.haiku_load

    cfg, params = haiku_load(str(path))
    net = make_counts_net_fn(cfg)

    def frac(d, angle):
        d = np.atleast_1d(np.asarray(d, float))
        angle = np.atleast_1d(np.asarray(angle, float))
        x = jnp.array(np.stack([np.log10(d), angle], 1))
        return np.power(10.0, np.asarray(net.apply(params, x))).reshape(-1)

    return cfg, frac


def _pf(ok):
    return "PASS" if ok else "FAIL"


def check_model(counts_path: Path, counts_data: Path = None, d_max_eval: float = 300.0) -> bool:
    cfg, frac = load_counts(counts_path)
    angles = np.radians(np.arange(0.0, 180.1, 2.5))
    dist = np.array([2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 100, 130, 160, 200, 250, 300], float)
    dist = dist[dist <= d_max_eval]
    print(f"\n=== {counts_path.name} ===")
    print(f"  config: { {k: v for k, v in cfg.items() if not hasattr(v, 'shape')} }")
    results = {}

    # 1. geometric bound
    worst = 0.0
    worst_d = None
    for d in dist:
        f = frac(np.full(angles.size, d), angles)
        ratio = f.max() * d * d / (np.pi * DOM_RADIUS_M**2)
        if ratio > worst:
            worst, worst_d = ratio, d
    results["geometric bound"] = worst <= 1.0
    print(
        f"  [1] geometric bound   : worst max_angle(frac) / (pi R^2 / d^2) = {worst:.3g} "
        f"at {worst_d:.0f} m  -> {_pf(results['geometric bound'])}"
    )

    # 2. cone convention
    bad = []
    for d in dist[(dist >= 5) & (dist <= 100)]:
        f = frac(np.full(angles.size, d), angles)
        peak = np.degrees(angles[np.argmax(f)])
        # A flat profile (isotropic source, equal at every angle within
        # 0.1 dex) is the correct viewing-angle table for such a source.
        flat = (np.log10(f.max()) - np.log10(max(f.min(), 1e-300))) < 0.1
        ok = flat or ((20.0 <= peak <= 65.0) and f[0] < f.max() and f[-1] < f.max())
        if not ok:
            bad.append((d, peak))
    results["cone convention"] = not bad
    shown = [(round(d), round(p)) for d, p in bad][:6] if bad else "none"
    print(
        "  [2] cone convention   : flat (isotropic) or peak in [20, 65] deg for 5-100 m; "
        f"violations {shown}  -> {_pf(results['cone convention'])}"
    )

    # 3. attenuation
    lam_ref = spectrum_weighted_absorption()
    dd = dist[(dist >= 40)]
    d2f = np.array([np.mean(frac(np.full(angles.size, d), angles)) * d * d for d in dd])
    monotone = bool(np.all(np.diff(d2f) < 0))
    slope = np.polyfit(dd, np.log(np.maximum(d2f, 1e-300)), 1)[0]
    lam_eff = -1.0 / slope if slope < 0 else float("inf")
    results["attenuation"] = monotone and (0.5 * lam_ref <= lam_eff <= 1.5 * lam_ref)
    print(
        f"  [3] attenuation       : d^2 x frac monotone beyond 40 m: {monotone}; "
        f"lambda_eff = {lam_eff:.0f} m vs spectrum-weighted absorption {lam_ref:.0f} m "
        f"(accept 0.5-1.5x)  -> {_pf(results['attenuation'])}"
    )

    # 4. training domain
    tmin, tmax = cfg.get("train_d_min"), cfg.get("train_d_max")
    if tmin is None or tmax is None:
        results["training domain"] = None
        print(
            "  [4] training domain   : not recorded in the pickle -> UNKNOWN "
            "(record train_d_min / train_d_max when training)"
        )
    else:
        results["training domain"] = tmax >= d_max_eval
        tag = "PASS" if results["training domain"] else "FAIL (extrapolation)"
        print(
            f"  [4] training domain   : trained over {tmin:.2f}-{tmax:.1f} m; "
            f"evaluation needs up to {d_max_eval:.0f} m  -> {tag}"
        )

    # 5. training rows
    if counts_data is not None and counts_data.exists():
        c = np.load(counts_data)
        d = 10 ** c["log10_dist"]
        a = c["angle"]
        s = c["log10_survival"]
        nd = c["n_detected"]
        ne = c["n_effective"] if "n_effective" in c.files else nd
        # A model fitted to pooled labels is compared with those labels. A raw
        # row far out is one or two heavy photons and says little on its own;
        # check 6 compares the raw rows where they can be trusted, in bands.
        fitted = {key: c[key] for key in c.files}
        pooled = cfg.get("label_pooling") == "fixed-window"
        if pooled:
            from hyperion.models.photon_arrival_time_nflow.training_data import pool_counts_rows

            fitted = pool_counts_rows(fitted)
        pred = np.log10(np.maximum(frac(d, a), 1e-300))
        m = nd >= 2
        rms = np.sqrt(np.mean((pred[m] - fitted["log10_survival"][m]) ** 2))
        rms = float(rms) if m.any() else float("nan")
        results["training rows"] = rms < 0.3
        kind = "pooled" if pooled else "raw"
        print(
            f"  [5] training rows     : {s.size} rows, {int((nd == 0).sum())} zero-hit "
            f"placeholders; distances {np.unique(np.round(d, 1)).size}, {d.min():.2f}-"
            f"{d.max():.1f} m; RMS(pred - {kind} labels) over >=2-hit rows = {rms:.2f} dex "
            f"(accept < 0.3)  -> {_pf(results['training rows'])}"
        )

        # 6. light conservation: the propagator multiplies the fraction by every
        # photon a source emits, so the network's solid-angle average over the
        # viewing angle must match that of the labels. The labels are noisy far
        # out, so distances are compared in bands, with room for the band's
        # statistical error. A network that loses light fails this.
        fine = np.linspace(0.0, np.pi, 3601)
        rows = []
        for ld in np.unique(c["log10_dist"]):
            r = (c["log10_dist"] == ld) & (nd > 0)
            if r.sum() < 3 or np.ptp(a[r]) < 0.5 * np.pi:
                continue
            o = np.argsort(a[r])
            th, lab, eff = a[r][o], 10 ** s[r][o], np.maximum(ne[r][o], 1e-9)
            q = np.gradient(th) * np.sin(th)
            q = q / q.sum()
            net = frac(np.full(fine.size, 10**ld), fine)
            rows.append(
                (
                    10**ld,
                    np.trapezoid(net * np.sin(fine), fine) / 2.0,
                    np.sum(q * lab),
                    np.sum((q * lab) ** 2 / eff),
                )
            )
        rows = np.array(rows).reshape(-1, 4)
        passed = rows.size > 0
        print("  [6] light conservation: network / labels in bands of 5 distances")
        for i in range(0, len(rows), 5):
            dd, net, lab, var = rows[i : i + 5].T
            ratio = np.sum(dd**2 * net) / np.sum(dd**2 * lab)
            sigma = np.sqrt(np.sum(dd**4 * var)) / np.sum(dd**2 * lab)
            ok = abs(np.log(ratio)) <= np.log(1.25) + 2.0 * sigma
            passed &= bool(ok)
            print(
                f"        {dd[0]:7.2f}-{dd[-1]:7.2f} m: {ratio:5.2f} +- {sigma:4.2f} "
                f"(accept 0.8-1.25 widened by 2 sigma)  -> {_pf(ok)}"
            )
        results["light conservation"] = passed

    verdict = all(v for v in results.values() if v is not None)
    print(
        f"  VERDICT: {'PASS' if verdict else 'FAIL'}"
        + ("" if results.get("training domain") is not None else "  (training domain unknown)")
    )
    return verdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", nargs="+", required=True, help="Counts-model pickle(s).")
    p.add_argument("--counts-data", default=None, help="counts_data.npz the model was trained on.")
    p.add_argument(
        "--d-max",
        type=float,
        default=300.0,
        help="Largest module distance the detector will need [m].",
    )
    args = p.parse_args()
    ok = True
    for c in args.counts:
        ok &= check_model(Path(c), Path(args.counts_data) if args.counts_data else None, args.d_max)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
