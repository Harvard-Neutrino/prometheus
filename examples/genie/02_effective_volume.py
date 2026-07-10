#!/usr/bin/env python3
"""10_effective_volume_genie.py
Estimate the effective detection volume for GENIE-injected events.

Uses a layered simulation strategy that exploits the approximate translational
symmetry of KM3NeT-style detectors.  Strings at similar radii see similar
photon yields, so simulating events around a single representative string per
radial layer and scaling by the string count gives a good estimate of V_eff
without running the full detector footprint.

Algorithm
---------
1. Load the detector geometry and extract the (x, y) position of every string.
2. Bin strings into ``n_layers`` equal-count radial layers.
3. For each layer, pick a representative string (closest to the layer's median
   radius) and generate vertex positions uniformly inside a Voronoi cell
   cylinder centred on that string.
4. Run Prometheus inject + propagate for those vertices using the full detector.
   Events are simulated in two stages: a pilot batch per layer, then the
   remaining event budget is allocated where it reduces the total V_eff
   uncertainty the most (weight ∝ n_strings × √(eff (1 − eff))).  Pass
   ``--no-adaptive`` for a flat allocation.
5. Apply a photon-hit threshold; compute the per-layer detection efficiency.
6. Scale to the full detector:

       V_eff = sum_layer (n_strings_layer × eff_layer × V_cell)

   where ``V_cell = π R_det² H / n_strings`` is the mean Voronoi cell volume
   and ``R_det = R_outer + d_nn / 2`` extends the outermost string radius by
   half the mean nearest-neighbour string spacing to include the detector
   volume beyond the last string ring.

Notes
-----
*  The photon hit counts come directly from the Olympus normalising-flow model
   before any DOM response (QE ≈ 25 %).  A raw photon threshold of 5 corresponds
   to roughly 1–2 expected detected PEs.  Increase ``--min-hits`` for stricter
   cuts.
*  Per-layer efficiency errors are Wilson intervals (z = 1), so layers with
   0 or 100 % detected events still carry a finite uncertainty.
*  Edge effects at the outermost layer are included in the outer-layer efficiency;
   the result is therefore a modest *underestimate* relative to a full detector
   Monte Carlo.

Usage
-----
Run from the repository root::

    .prometheus_env/bin/python examples/10_effective_volume_genie.py

    # Custom file, geometry, and statistics
    .prometheus_env/bin/python examples/10_effective_volume_genie.py \\
        --file tests/resources/genie_example.root \\
        --geo resources/geofiles/arca.geo \\
        --n-events 500 --n-layers 3 --min-hits 5
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from time import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    from prometheus import Prometheus, config
    from prometheus.detector import detector_from_geo
    from prometheus.utils.layered_sim import build_layers, run_batch, wilson_sigma
except Exception:
    logger.exception("Error importing Prometheus.")
    logger.info("Hint: source scripts/activate.sh .prometheus_env")
    sys.exit(1)

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_compilation_cache_dir",
                      str(Path(__file__).resolve().parent.parent.parent / ".jax_cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_FILE = REPO_ROOT / "tests" / "resources" / "genie_example.root"
_DEFAULT_GEO  = REPO_ROOT / "resources" / "geofiles" / "arca.geo"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate effective detection volume for GENIE-injected events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", default=str(_DEFAULT_FILE), metavar="ROOT",
                   help="GENIE gRooTracker ROOT file to sample from.")
    p.add_argument("--geo", default=str(_DEFAULT_GEO), metavar="GEO",
                   help="Detector geometry file.")
    p.add_argument("--n-events", type=int, default=200, metavar="N",
                   help="Mean events per layer; the total budget is "
                        "n-events × n-layers.")
    p.add_argument("--n-layers", type=int, default=3, metavar="N",
                   help="Number of radial layers to partition strings into.")
    p.add_argument("--no-adaptive", action="store_true",
                   help="Disable adaptive event allocation and simulate exactly "
                        "n-events per layer in a single batch.")
    p.add_argument("--min-hits", type=int, default=5, metavar="N",
                   help="Minimum total raw photon hits to count an event as detected.")
    p.add_argument("--min-modules", type=int, default=1, metavar="N",
                   help="Minimum number of distinct modules hit to count an event as detected.")
    p.add_argument("--max-distance", type=float, default=100.0, metavar="M",
                   help="Maximum source–module distance passed to the Olympus propagator [m].")
    p.add_argument("--seed", type=int, default=42, metavar="N",
                   help="Base random seed for vertex placement.")
    p.add_argument("--run-number", type=int, default=10, metavar="N",
                   help="Run number embedded in output parquet filenames.")
    p.add_argument("--out", default=str(REPO_ROOT / "output" / "10_effective_volume_genie.csv"),
                   metavar="FILE",
                   help="Output CSV file for per-layer results.")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="Directory for per-layer parquet files. "
                        "Defaults to the same directory as --out.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()

    genie_file = Path(args.file)
    geo_file   = Path(args.geo)

    for path, label in [(genie_file, "GENIE"), (geo_file, "geo")]:
        if not path.exists():
            logger.error("%s file not found: %s", label, path)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Detector geometry and layer partitioning
    # ------------------------------------------------------------------
    detector = detector_from_geo(str(geo_file))
    layers, cell = build_layers(detector, args.n_layers)

    print(f"Geo file    : {geo_file.name}")
    print(f"Strings     : {cell.n_strings}  |  Modules: {len(detector.module_coords)}")
    print(f"Outer radius: {cell.R_outer:.1f} m (+{0.5 * cell.d_nn:.1f} m skirt)  "
          f"|  Height: {cell.H:.1f} m")
    print(f"Cell radius : {cell.r_cell:.1f} m  |  V_cell: {cell.V_cell:.2e} m³")

    print(f"\nLayers ({args.n_layers}):")
    for i, lay in enumerate(layers):
        print(f"  Layer {i}: {lay['n_strings']:4d} strings  "
              f"r = {lay['r_range'][0]:6.1f}–{lay['r_range'][1]:6.1f} m  "
              f"rep = ({lay['rep_xy'][0]:+.1f}, {lay['rep_xy'][1]:+.1f})")

    # ------------------------------------------------------------------
    # Prometheus initialisation (once — loads JAX models)
    # ------------------------------------------------------------------
    out_path   = Path(args.out)
    parquet_dir = Path(args.out_dir) if args.out_dir else out_path.parent
    parquet_dir.mkdir(parents=True, exist_ok=True)
    channel = genie_file.stem  # used to label output parquet filenames

    config.run.run_number                               = args.run_number
    config.run.random_state_seed                        = args.seed
    config.detector.geo_file                            = str(geo_file)
    config.injection.name                               = "GENIE"
    config.injection.genie.paths.injection_file         = str(genie_file)
    config.injection.genie.inject                       = True
    config.injection.genie.simulation.placement         = "fixed"
    config.photon_propagator.olympus.simulation.max_distance = args.max_distance

    print("\nInitialising Prometheus …")
    t0 = time()
    prom = Prometheus(detector=detector)
    print(f"  done in {time() - t0:.1f} s")

    # ------------------------------------------------------------------
    # Per-layer simulation
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    total_budget = args.n_events * args.n_layers
    if args.no_adaptive:
        n_pilot = args.n_events
    else:
        n_pilot = min(args.n_events, max(30, args.n_events // 5))

    # Stage 1 — pilot batches with equal statistics per layer.
    print(f"\nStage 1 — pilot ({n_pilot} events per layer)")
    layer_stats = []
    batch_idx = 0
    for i, lay in enumerate(layers):
        parquet_file = (
            parquet_dir / f"signal_{channel}_layer{i}_pilot_r{args.run_number}.parquet"
        )
        t_batch = time()
        hit_counts, mod_counts = run_batch(
            prom, config, rng, lay["rep_xy"], n_pilot, cell,
            args.seed + batch_idx, parquet_file,
        )
        batch_idx += 1

        detected = (hit_counts >= args.min_hits) & (mod_counts >= args.min_modules)
        layer_stats.append({
            "n": n_pilot,
            "k": int(detected.sum()),
            "hits": [hit_counts],
            "mods": [mod_counts],
        })
        print(f"  Layer {i}: {layer_stats[i]['k']:4d}/{n_pilot} detected  "
              f"[{time() - t_batch:.0f} s]  → {parquet_file.name}")

    # Stage 2 — spend the remaining budget where it shrinks the total V_eff
    # error the most.  A layer's contribution to sigma_V_eff scales with
    # n_strings × sqrt(eff (1 - eff) / n), so allocate events proportional to
    # n_strings × sqrt(p (1 - p)).  The Bayesian mean (k+1)/(n+2) keeps the
    # weight finite when a pilot batch detected none or all of its events.
    if not args.no_adaptive:
        remaining = total_budget - n_pilot * args.n_layers
        if remaining > 0:
            p_bayes = np.array([(s["k"] + 1) / (s["n"] + 2) for s in layer_stats])
            weights = np.array([lay["n_strings"] for lay in layers]) * np.sqrt(
                p_bayes * (1.0 - p_bayes)
            )
            n_main = np.rint(remaining * weights / weights.sum()).astype(int)

            print(f"\nStage 2 — adaptive top-up ({remaining} events): "
                  + ", ".join(f"layer {i}: {n}" for i, n in enumerate(n_main)))
            for i, (lay, n_ev) in enumerate(zip(layers, n_main)):
                if n_ev <= 0:
                    continue
                parquet_file = (
                    parquet_dir / f"signal_{channel}_layer{i}_main_r{args.run_number}.parquet"
                )
                t_batch = time()
                hit_counts, mod_counts = run_batch(
                    prom, config, rng, lay["rep_xy"], int(n_ev), cell,
                    args.seed + batch_idx, parquet_file,
                )
                batch_idx += 1

                detected = (hit_counts >= args.min_hits) & (mod_counts >= args.min_modules)
                s = layer_stats[i]
                s["n"] += int(n_ev)
                s["k"] += int(detected.sum())
                s["hits"].append(hit_counts)
                s["mods"].append(mod_counts)
                print(f"  Layer {i}: {int(detected.sum()):4d}/{int(n_ev)} detected  "
                      f"[{time() - t_batch:.0f} s]  → {parquet_file.name}")

    # ------------------------------------------------------------------
    # Per-layer efficiencies and effective volumes
    # ------------------------------------------------------------------
    results = []
    for i, (lay, s) in enumerate(zip(layers, layer_stats)):
        hit_counts = np.concatenate(s["hits"])
        mod_counts = np.concatenate(s["mods"])
        eff       = s["k"] / s["n"]
        sigma_eff = wilson_sigma(s["k"], s["n"])

        V_eff_layer       = lay["n_strings"] * eff * cell.V_cell
        sigma_V_eff_layer = lay["n_strings"] * sigma_eff * cell.V_cell

        print(f"\nLayer {i} — {lay['n_strings']} strings  "
              f"r = {lay['r_range'][0]:.0f}–{lay['r_range'][1]:.0f} m")
        print(f"  detected : {s['k']}/{s['n']}  eff = {eff:.4f} ± {sigma_eff:.4f}")
        print(f"  V_eff    : {V_eff_layer:.3e} ± {sigma_V_eff_layer:.3e} m³  "
              f"({V_eff_layer/1e9:.5f} km³)")

        results.append({
            "layer":              i,
            "n_strings":          lay["n_strings"],
            "r_min_m":            lay["r_range"][0],
            "r_max_m":            lay["r_range"][1],
            "rep_x_m":            lay["rep_xy"][0],
            "rep_y_m":            lay["rep_xy"][1],
            "n_events":           s["n"],
            "n_pilot":            n_pilot,
            "n_detected":         s["k"],
            "eff":                eff,
            "sigma_eff":          sigma_eff,
            "V_eff_m3":           V_eff_layer,
            "sigma_V_eff_m3":     sigma_V_eff_layer,
            "median_hits":        float(np.median(hit_counts)),
            "median_modules":     float(np.median(mod_counts)),
        })

    # ------------------------------------------------------------------
    # Total effective volume
    # ------------------------------------------------------------------
    V_eff_total       = sum(r["V_eff_m3"]       for r in results)
    sigma_V_eff_total = np.sqrt(sum(r["sigma_V_eff_m3"]**2 for r in results))

    print("\n" + "=" * 60)
    print("EFFECTIVE VOLUME SUMMARY")
    print("=" * 60)
    print(f"  Geo          : {geo_file.name}")
    print(f"  GENIE        : {genie_file.name}")
    print(f"  Threshold    : ≥ {args.min_hits} photon hits, ≥ {args.min_modules} modules")
    print(f"  max_distance : {args.max_distance} m")
    print()
    header = (f"{'Layer':>5}  {'n_str':>6}  {'r [m]':>13}  {'n_ev':>6}  "
              f"{'eff':>8}  {'V_eff [m³]':>14}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"  {r['layer']:3d}    {r['n_strings']:4d}  "
              f"{r['r_min_m']:5.0f}–{r['r_max_m']:5.0f}  {r['n_events']:6d}  "
              f"{r['eff']:7.4f}  {r['V_eff_m3']:14.3e}")
    print("-" * len(header))
    print(f"  Total V_eff = {V_eff_total:.3e} ± {sigma_V_eff_total:.3e} m³")
    print(f"             = {V_eff_total/1e9:.5f} ± {sigma_V_eff_total/1e9:.5f} km³")
    print("=" * 60)

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys()) + ["V_cell_m3", "r_cell_m", "R_outer_m", "R_det_m",
                                             "d_nn_m", "H_m",
                                             "min_hits_threshold", "min_modules_threshold",
                                             "max_distance_m",
                                             "V_eff_total_m3", "sigma_V_eff_total_m3",
                                             "geo_file", "genie_file"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row.update({
                "V_cell_m3":              cell.V_cell,
                "r_cell_m":               cell.r_cell,
                "R_outer_m":              cell.R_outer,
                "R_det_m":                cell.R_det,
                "d_nn_m":                 cell.d_nn,
                "H_m":                    cell.H,
                "min_hits_threshold":     args.min_hits,
                "min_modules_threshold":  args.min_modules,
                "max_distance_m":         args.max_distance,
                "V_eff_total_m3":         V_eff_total,
                "sigma_V_eff_total_m3":   sigma_V_eff_total,
                "geo_file":               geo_file.name,
                "genie_file":             genie_file.name,
            })
            writer.writerow(row)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
