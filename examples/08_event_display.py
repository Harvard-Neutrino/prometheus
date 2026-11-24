#!/usr/bin/env python3
"""08_event_display.py
Write an interactive 3-D display of one multi-PMT event as a single HTML file.

The style follows KM3NeT's RainbowAlga.jl: each photon is a small sphere on the
axis of the PMT it hit, later photons farther out, colored by arrival time. The
line is the muon, the dashes the neutrino, and the cone the Cherenkov front.
Clicking a module shows its PMTs. The file needs a browser with WebGL 2 and no
network access.

The muon is drawn until it leaves the detector region, because its stopping point
is not in the output file. Modules are enlarged by --scale.

Usage
-----
python examples/08_event_display.py examples/output/numu_lom/1_photons.parquet \\
    --geo examples/output/numu_lom/detector.geo
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prometheus.detector import detector_from_geo

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "event_display.html"
LOM16_CONF = HERE.parent / "resources" / "optical_modules" / "lom16" / "om.conf"
SPEED_OF_LIGHT = 0.299792458  # m / ns
PHASE_INDEX = 1.32  # ice, sets the Cherenkov angle drawn

# The module geometry is shared with the static event view.
_spec = importlib.util.spec_from_file_location("event_view", HERE / "04_event_view.py")
event_view = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(event_view)

NAMES = {12: "electron", 14: "muon", 16: "tau"}


def module_records(detector, photons):
    """Return every module with its photons, sorted by time."""
    records = {}
    for module in detector.modules:
        key = tuple(int(i) for i in module.key)
        records[key] = {
            "string": key[0],
            "om": key[1],
            "pos": [round(float(x), 3) for x in module.pos],
            "hits": [],
        }
    keys = list(zip(photons["string_id"], photons["sensor_id"]))
    for key, t, pmt in zip(keys, photons["t"], photons["pmt_id"]):
        if key not in records:
            raise SystemExit(f"Module {key} has photons but is not in the geo file")
        if pmt is None:
            raise SystemExit("Photons have no PMT; was PPC run in multi-PMT mode?")
        records[key]["hits"].append([round(t, 2), int(pmt)])
    zenith, azimuth = np.array(photons["photon_zenith"]), np.array(photons["photon_azimuth"])
    travel = event_view.unit_vector(zenith, azimuth)
    index = {}
    for i, key in enumerate(keys):
        index.setdefault(key, []).append(i)
    for key, rows in index.items():
        records[key]["hits"].sort()
        mean = travel[rows].mean(axis=0)
        length = np.linalg.norm(mean)
        if length > 0:
            records[key]["travel"] = [float(x) for x in mean / length]
    return list(records.values())


def track_length(vertex, direction, positions, margin=60.0):
    """Return the distance from the vertex to the edge of the detector region."""
    center = (positions.max(axis=0) + positions.min(axis=0)) / 2
    radius = np.linalg.norm(positions - center, axis=1).max() + margin
    offset = vertex - center
    b = offset @ direction
    discriminant = b**2 - offset @ offset + radius**2
    if discriminant <= 0:
        return radius
    return max(float(-b + np.sqrt(discriminant)), margin)


def build(row, detector, om, scale):
    """Return the data the HTML template needs."""
    photons, truth = row["photons"], row["mc_truth"]
    for name in ("pmt_id", "photon_zenith"):
        if name not in photons:
            raise SystemExit(f"The parquet file has no {name}; use output_mode 'standard'")
    if not photons["t"]:
        raise SystemExit("This event has no photons")
    records = module_records(detector, photons)
    positions = np.array([r["pos"] for r in records])
    if max(photons["pmt_id"]) >= om["n_pmts"]:
        raise SystemExit(f"pmt_id does not fit a module with {om['n_pmts']} PMTs")

    muons = [i for i, pdg in enumerate(truth["final_state_type"]) if abs(pdg) == 13]
    if not muons:
        raise SystemExit("This event has no muon in the final state")
    i = muons[0]
    vertex = np.array([truth[f"initial_state_{axis}"] for axis in "xyz"])
    direction = event_view.unit_vector(
        np.array(truth["final_state_zenith"][i]), np.array(truth["final_state_azimuth"][i])
    )

    zenith, azimuth = np.radians(np.array(om["pmt_dirs"], dtype=float)).T
    axes = event_view.unit_vector(zenith, azimuth)
    radius, height = event_view.profile(om, "vessel", n=8)
    pmt_radius = 0.051
    times = np.array(photons["t"])
    t_min, t_max = np.percentile(times, [0, 95])
    flavor = NAMES.get(abs(truth["initial_state_type"]), "unknown").capitalize()
    kind = "antineutrino" if truth["initial_state_type"] < 0 else "neutrino"
    n_hit = sum(bool(r["hits"]) for r in records)
    return {
        "title": f"{flavor} {kind}, {truth['initial_state_energy'] / 1e3:.1f} TeV",
        "lines": [
            f"Muon {truth['final_state_energy'][i] / 1e3:.1f} TeV, "
            f"zenith {np.degrees(truth['final_state_zenith'][i]):.0f} deg (travel direction)",
            f"{len(times)} photons on {n_hit} of {len(records)} modules, "
            f"{len({r['string'] for r in records})} strings",
            f"Modules enlarged {scale:g} times. Click a module for its PMTs. "
            "Drag to rotate, shift-drag to pan, scroll to zoom.",
        ],
        "modules": records,
        "selected": int(np.argmax([len(r["hits"]) for r in records])),
        "pmt_axes": axes.tolist(),
        "pmt_distance": event_view.surface_distance(om, "vessel", axes).tolist(),
        "pmt_radius": pmt_radius,
        "pmt_depth": pmt_radius**2 / (2 * om["Rr"]),
        "outline": np.column_stack([radius, height]).tolist(),
        "scale": scale,
        "vertex": vertex.tolist(),
        "direction": direction.tolist(),
        "track_length": track_length(vertex, direction, positions),
        "c": SPEED_OF_LIGHT,
        "cherenkov_angle": float(np.arccos(1 / PHASE_INDEX)),
        "t_min": float(t_min),
        "t_max": float(t_max if t_max > t_min else t_min + 1),
        "t_end": float(np.percentile(times, 99)),
        # Look across the track.
        "camera_az": float(np.arctan2(direction[1], direction[0]) - np.pi / 2 + 0.25),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("parquet", type=Path, help="Prometheus photon parquet file")
    parser.add_argument("--geo", required=True, help="geo file with all modules")
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--om-conf", default=str(LOM16_CONF))
    parser.add_argument("--om-type", type=int, default=171)
    parser.add_argument("--scale", type=float, default=15.0, help="module enlargement")
    parser.add_argument("--out", type=Path, help="HTML file (default: beside the parquet file)")
    args = parser.parse_args()

    rows = pq.read_table(args.parquet).to_pylist()
    if not 0 <= args.event < len(rows):
        parser.error(f"--event must be between 0 and {len(rows) - 1}")
    om = event_view.load_om(args.om_conf, args.om_type)
    data = build(rows[args.event], detector_from_geo(args.geo), om, args.scale)
    # "<" must not end the script element early.
    text = json.dumps(data).replace("<", "\\u003c")
    out = args.out or args.parquet.with_name("event_display.html")
    out.write_text(TEMPLATE.read_text().replace("__EVENT_DATA__", text))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
