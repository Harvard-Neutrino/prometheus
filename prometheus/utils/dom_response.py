# -*- coding: utf-8 -*-
# dom_response.py
# DOM / mDOM detector response shared by the pulse-conversion examples.

"""Optical-module response model: photons to digitised FADC pulses.

Converts raw photon arrival times at a module into per-PMT digitised pulses,
via a KM3NeT-style mDOM response model:

1. Quantum efficiency (QE) binomial filter
2. Photon assignment to individual mDOM PMTs (Lambert cosine-law visibility)
   -- :mod:`prometheus.utils.pmt_response`
3. Transit-time spread (TTS) Gaussian smearing
4. Dark noise injection (thermal + correlated radioactive bursts)
5. SPE-template convolution and FADC digitisation (3.3 ns bins)
   -- :mod:`prometheus.utils.fadc_digitization`

This module keeps the top-level orchestrator, :func:`process_event`, and
re-exports the building blocks from ``pmt_response``/``fadc_digitization``
(plus their constants, now sourced from
:class:`prometheus.config_types.DOMResponseConfig`) so existing
``from prometheus.utils.dom_response import X`` call sites keep working.
"""

from collections import defaultdict

import numpy as np

from prometheus.config_types import DOMResponseConfig
from prometheus.utils.fadc_digitization import (  # noqa: F401
    FADC_BIN_NS,
    PMT_DARK_RATE_HZ,
    PULSE_WIDTH_NS,
    SIM_DT_NS,
    SPE_MEAN,
    SPE_SIGMA,
    TOT_MAX_NS,
    TOT_THRESHOLD_PE,
    TTS_NS,
    _threshold_crossings,
    dark_noise,
    generate_fadc_response,
)
from prometheus.utils.pmt_response import (  # noqa: F401
    EC_WATER_GEV,
    LAMBDA_I_WATER_M,
    MUON_DEDX_GEV_PER_M,
    MUON_MASS_GEV,
    N_PMTS,
    PMT_DIRS,
    X0_WATER_M,
    assign_to_pmts,
    assign_to_pmts_per_hit,
    emission_length,
    fibonacci_sphere,
)

QE = DOMResponseConfig().qe  # total module quantum efficiency


def process_event(
    photons: dict,
    vertex_pos: np.ndarray,
    rng: np.random.Generator,
    qe: float = QE,
    dark_rate_hz: float = PMT_DARK_RATE_HZ,
    source_points: np.ndarray = None,
    tot_threshold_pe: float = TOT_THRESHOLD_PE,
    tot_max_ns: float = TOT_MAX_NS,
) -> dict:
    """Group photon hits by module, assign to individual PMTs, build FADC pulses.

    Parameters
    ----------
    photons : dict
        The ``photons`` field from one parquet row (keys ``string_id``,
        ``sensor_id``, ``t``, ``sensor_pos_x/y/z``).
    vertex_pos : ndarray, shape (3,)
        Event vertex position [m], used as the photon source when
        ``source_points`` is not given.
    rng : numpy.random.Generator
        Random number generator.
    qe : float
        Total module quantum efficiency.
    dark_rate_hz : float
        Per-PMT dark-noise rate [Hz].
    source_points : ndarray, shape (n_hits, 3), optional
        Per-photon emission points (e.g. the shower maximum of the particle
        that produced each hit, see :func:`emission_length`).  When given,
        every photon is assigned to PMTs using its own arrival direction,
        which preserves the event topology in the intra-module hit pattern.
    tot_threshold_pe : float
        Discriminator threshold for ToT hit extraction, PE-equivalent
        amplitude.
    tot_max_ns : float
        ToT saturation cap [ns].

    Returns
    -------
    dict
        One entry per *PMT* that fired, with keys ``string_id``, ``sensor_id``,
        ``sensor_pos_x/y/z``, ``pmt_id``, ``pmt_dir_x/y/z``, ``n_pe``,
        ``fadc_t``, ``fadc_q``, ``hit_t``, ``tot_ns``.  ``hit_t``/``tot_ns``
        are the ToT hits (leading-edge time, ToT duration) a real KM3NeT DOM
        front-end would report; ``fadc_t``/``fadc_q`` are the idealized
        analog charge readout, retained for debugging/visualization only.
    """
    string_ids = np.asarray(photons["string_id"])
    sensor_ids = np.asarray(photons["sensor_id"])
    times      = np.asarray(photons["t"], dtype=float)
    mod_pos    = np.column_stack([
        np.asarray(photons["sensor_pos_x"], dtype=float),
        np.asarray(photons["sensor_pos_y"], dtype=float),
        np.asarray(photons["sensor_pos_z"], dtype=float),
    ]) if len(times) else np.empty((0, 3))

    if source_points is None:
        source_points = np.broadcast_to(
            np.asarray(vertex_pos, dtype=float), (len(times), 3)
        )
    else:
        source_points = np.asarray(source_points, dtype=float)

    # Group by module
    module_hits: dict[tuple, dict] = defaultdict(
        lambda: {"times": [], "points": [], "pos": None}
    )
    for sid, mid, t, pos, point in zip(string_ids, sensor_ids, times, mod_pos,
                                       source_points):
        key = (int(sid), int(mid))
        module_hits[key]["times"].append(float(t))
        module_hits[key]["points"].append(point)
        module_hits[key]["pos"] = pos

    out: dict[str, list] = {
        "string_id": [], "sensor_id": [],
        "sensor_pos_x": [], "sensor_pos_y": [], "sensor_pos_z": [],
        "pmt_id": [],
        "pmt_dir_x": [], "pmt_dir_y": [], "pmt_dir_z": [],
        "n_pe": [], "fadc_t": [], "fadc_q": [],
        "hit_t": [], "tot_ns": [],
    }

    for (sid, mid), info in module_hits.items():
        mod_centre = info["pos"]

        # Per-photon unit vectors from the module toward the emission points
        diffs = np.asarray(info["points"]) - mod_centre
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        source_dirs = np.divide(diffs, norms,
                                out=np.tile([[0.0, 0.0, 1.0]], (len(diffs), 1)),
                                where=norms > 0)

        # Assign photons to individual PMTs
        pmt_hits = assign_to_pmts_per_hit(
            np.array(info["times"]), source_dirs, PMT_DIRS, qe, rng
        )

        for pmt_idx, hit_times in pmt_hits.items():
            # Per-PMT FADC + ToT (qe=1 since assignment already applied QE)
            fadc_t, fadc_q, n_pe, hit_t, tot_ns = generate_fadc_response(
                hit_times, qe=1.0, dark_rate_hz=dark_rate_hz, rng=rng,
                tot_threshold_pe=tot_threshold_pe, tot_max_ns=tot_max_ns,
            )
            pmt_dir = PMT_DIRS[pmt_idx]
            out["string_id"].append(sid)
            out["sensor_id"].append(mid)
            out["sensor_pos_x"].append(float(mod_centre[0]))
            out["sensor_pos_y"].append(float(mod_centre[1]))
            out["sensor_pos_z"].append(float(mod_centre[2]))
            out["pmt_id"].append(pmt_idx)
            out["pmt_dir_x"].append(float(pmt_dir[0]))
            out["pmt_dir_y"].append(float(pmt_dir[1]))
            out["pmt_dir_z"].append(float(pmt_dir[2]))
            out["n_pe"].append(n_pe)
            out["fadc_t"].append(fadc_t.tolist())
            out["fadc_q"].append(fadc_q.tolist())
            out["hit_t"].append(hit_t.tolist())
            out["tot_ns"].append(tot_ns.tolist())
    return out
