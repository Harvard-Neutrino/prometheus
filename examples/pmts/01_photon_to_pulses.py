#!/usr/bin/env python3
"""07_photon_to_pulses.py
Convert Prometheus photon hits to digitised FADC pulses.

Loads ``output/10_photons.parquet`` produced by an upstream Prometheus run
(e.g. example 05), groups photon arrival times by module (string_id,
sensor_id), and applies a KM3NeT-style mDOM response model:

  1. Quantum efficiency (QE) binomial filter
  2. Transit-time spread (TTS) Gaussian smearing
  3. Dark noise injection (thermal + correlated radioactive bursts)
  4. SPE-template convolution and FADC digitisation (3.3 ns bins)

Output ``output/11_pulses.parquet`` has one row per event, with per-module
fields ``string_id``, ``sensor_id``, ``n_pe``, ``fadc_t`` and ``fadc_q``.

Usage
-----
Run from the repository root::

    .prometheus_env/bin/python examples/07_photon_to_pulses.py
"""

import argparse
import logging
import sys
from pathlib import Path

import awkward as ak
import numpy as np

from prometheus.utils.dom_response import process_event

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = REPO_ROOT / "output" / "10_photons.parquet"
OUTPUT_FILE = REPO_ROOT / "output" / "11_pulses.parquet"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Convert photon hits to FADC pulses")
    parser.add_argument("--input", type=Path, default=INPUT_FILE,
                        help="Input parquet file (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Output parquet file (default: %(default)s)")
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)

    print(f"Loading {input_file}")
    events = ak.from_parquet(str(input_file))
    n_events = len(events)
    print(f"  {n_events} events")

    rng = np.random.default_rng(42)
    pulse_records: list[dict] = []

    for i in range(n_events):
        ev = events[i]
        photons   = ak.to_list(ev["photons"])
        mc        = ak.to_list(ev["mc_truth"])
        vertex_pos = np.array([
            mc["initial_state_x"],
            mc["initial_state_y"],
            mc["initial_state_z"],
        ])
        record = process_event(photons, vertex_pos, rng)
        pulse_records.append(record)
        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{n_events} events")

    out = ak.Array(pulse_records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(out, str(output_file))
    print(f"Saved pulses → {output_file}")

    total_pmt_hits = sum(len(r["pmt_id"]) for r in pulse_records)
    total_pe = sum(sum(r["n_pe"]) for r in pulse_records)
    print(f"  PMT hits (fired channels) : {total_pmt_hits}")
    print(f"  total signal PEs          : {total_pe}")


if __name__ == "__main__":
    main()
