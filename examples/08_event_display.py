#!/usr/bin/env python3
"""08_event_display.py
Write an interactive 3-D display of multi-PMT events as a single HTML file.

All events with photons and a muon are included; use the menu, the Previous and Next
buttons, or the arrow keys to move between them. --events selects some of them.

The style follows KM3NeT's RainbowAlga.jl: each photon is a small sphere on the
axis of the PMT it hit, later photons farther out, colored by arrival time. The
line is the muon, the dashes the neutrino, and the cone the Cherenkov front.
Clicking a module shows its PMTs. The
arrow there, and the thin line in the scene, are the path of unscattered Cherenkov
light from the muon to the module, or from the vertex if the muon would have had to
emit it before it was created. The file needs a browser with WebGL 2 and no
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
PMT_RADIUS = 0.051  # m, as drawn
PHASE_INDEX = 1.32  # ice, sets the Cherenkov angle drawn

# The module geometry is shared with the static event view.
_spec = importlib.util.spec_from_file_location("event_view", HERE / "04_event_view.py")
event_view = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(event_view)

NAMES = {12: "electron", 14: "muon", 16: "tau"}


def direct_light(position, vertex, direction, length, angle):
    """Return where unscattered light to ``position`` leaves the track.

    Parameters
    ----------
    position, vertex, direction : np.ndarray
        Module position, start of the track, and unit direction of the track.
    length : float
        Track length; light from beyond it is taken from the end of the track.
    angle : float
        Cherenkov angle [rad].

    Returns
    -------
    tuple
        Emission point and whether it is at the Cherenkov angle.
    """
    offset = position - vertex
    along = offset @ direction
    distance = np.linalg.norm(offset - along * direction)
    emission = along - distance / np.tan(angle)
    clipped = min(max(emission, 0.0), length)
    return vertex + clipped * direction, bool(clipped == emission)


def module_hits(index, photons):
    """Return the photons of each module, sorted by time, by position in ``index``."""
    hits = {}
    for string, om, t, pmt in zip(
        photons["string_id"], photons["sensor_id"], photons["t"], photons["pmt_id"]
    ):
        if (string, om) not in index:
            raise SystemExit(f"Module {(string, om)} has photons but is not in the geo file")
        if pmt is None:
            raise SystemExit("Photons have no PMT; was PPC run in multi-PMT mode?")
        hits.setdefault(index[(string, om)], []).append([round(t, 1), int(pmt)])
    for rows in hits.values():
        rows.sort()
    return hits


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


def rounded(vector, digits=3):
    """Return a list of rounded floats, to keep the file small."""
    return [round(float(x), digits) for x in vector]


def build_detector(detector, om, scale):
    """Return the data shared by all events."""
    zenith, azimuth = np.radians(np.array(om["pmt_dirs"], dtype=float)).T
    axes = event_view.unit_vector(zenith, azimuth)
    radius, height = event_view.profile(om, "vessel", n=8)
    modules = []
    for module in detector.modules:
        string, number = (int(i) for i in module.key)
        modules.append({"string": string, "om": number, "pos": rounded(module.pos)})
    return {
        "modules": modules,
        "pmt_axes": axes.tolist(),
        "pmt_distance": event_view.surface_distance(om, "vessel", axes).tolist(),
        "pmt_radius": PMT_RADIUS,
        "pmt_depth": PMT_RADIUS**2 / (2 * om["Rr"]),
        "outline": np.column_stack([radius, height]).tolist(),
        "scale": scale,
        "c": SPEED_OF_LIGHT,
        "cherenkov_angle": float(np.arccos(1 / PHASE_INDEX)),
    }


def find_muon(row):
    """Return the index of the primary muon in the final state, or None."""
    truth = row["mc_truth"]
    for i, pdg in enumerate(truth["final_state_type"]):
        if abs(pdg) == 13:
            return i
    return None


def build_event(row, number, modules, n_pmts):
    """Return the data of one event, or None if it has no photons or no muon."""
    photons, truth = row.get("photons"), row["mc_truth"]
    i = find_muon(row)
    if i is None or not photons or not photons["t"]:
        return None
    if "pmt_id" not in photons:
        raise SystemExit("The parquet file has no pmt_id; use output_mode 'standard'")
    index = {(m["string"], m["om"]): k for k, m in enumerate(modules)}
    hits = module_hits(index, photons)
    if max(photons["pmt_id"]) >= n_pmts:
        raise SystemExit(f"pmt_id does not fit a module with {n_pmts} PMTs")

    positions = np.array([m["pos"] for m in modules])
    vertex = np.array([truth[f"initial_state_{axis}"] for axis in "xyz"])
    direction = event_view.unit_vector(
        np.array(truth["final_state_zenith"][i]), np.array(truth["final_state_azimuth"][i])
    )
    length = track_length(vertex, direction, positions)
    angle = np.arccos(1 / PHASE_INDEX)
    records = {}
    for k, rows in hits.items():
        point, on_cone = direct_light(positions[k], vertex, direction, length, angle)
        record = {"hits": rows}
        path = positions[k] - point
        distance = np.linalg.norm(path)
        if distance > 0:
            record["emission"] = rounded(point)
            record["travel"] = rounded(path / distance, 5)
            record["distance"] = round(float(distance), 1)
            record["on_cone"] = on_cone
        records[k] = record

    times = np.array(photons["t"])
    t_min, t_max = np.percentile(times, [0, 95])
    flavor = NAMES.get(abs(truth["initial_state_type"]), "unknown").capitalize()
    kind = "antineutrino" if truth["initial_state_type"] < 0 else "neutrino"
    energy = truth["initial_state_energy"] / 1e3
    return {
        "number": number,
        "label": f"{number}: {energy:.1f} TeV, {len(times)} photons",
        "title": f"{flavor} {kind}, {energy:.1f} TeV",
        "lines": [
            f"Muon {truth['final_state_energy'][i] / 1e3:.1f} TeV, "
            f"zenith {np.degrees(truth['final_state_zenith'][i]):.0f} deg (travel direction)",
            f"{len(times)} photons on {len(hits)} of {len(modules)} modules, "
            f"{len({modules[k]['string'] for k in hits})} strings",
        ],
        "hits": records,
        "selected": max(hits, key=lambda k: len(hits[k])),
        "vertex": rounded(vertex),
        "direction": direction.tolist(),
        "track_length": length,
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
    parser.add_argument("--events", type=int, nargs="+", help="events to include (default: all)")
    parser.add_argument("--om-conf", default=str(LOM16_CONF))
    parser.add_argument("--om-type", type=int, default=171)
    parser.add_argument("--scale", type=float, default=15.0, help="module enlargement")
    parser.add_argument("--out", type=Path, help="HTML file (default: beside the parquet file)")
    args = parser.parse_args()

    rows = pq.read_table(args.parquet).to_pylist()
    numbers = range(len(rows)) if args.events is None else args.events
    for number in numbers:
        if not 0 <= number < len(rows):
            parser.error(f"events must be between 0 and {len(rows) - 1}")
    om = event_view.load_om(args.om_conf, args.om_type)
    data = build_detector(detector_from_geo(args.geo), om, args.scale)
    events = [build_event(rows[n], n, data["modules"], om["n_pmts"]) for n in numbers]
    data["events"] = [event for event in events if event is not None]
    if not data["events"]:
        raise SystemExit("No event has both photons and a muon")
    # "<" must not end the script element early.
    text = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    out = args.out or args.parquet.with_name("event_display.html")
    out.write_text(TEMPLATE.read_text().replace("__EVENT_DATA__", text))
    skipped = len(events) - len(data["events"])
    print(f"Saved {out}: {len(data['events'])} events, {skipped} without photons or muon")


if __name__ == "__main__":
    main()
