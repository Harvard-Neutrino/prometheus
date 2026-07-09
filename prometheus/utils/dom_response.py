# -*- coding: utf-8 -*-
# dom_response.py
# DOM / mDOM detector response shared by the pulse-conversion examples.

"""Optical-module response model: photons to digitised FADC pulses.

Converts raw photon arrival times at a module into per-PMT digitised pulses,
mirroring the model used in a reference DOM-response implementation:

1. Quantum efficiency (QE) binomial filter
2. Photon assignment to individual mDOM PMTs (Lambert cosine-law visibility)
3. Transit-time spread (TTS) Gaussian smearing
4. Dark noise injection (thermal + correlated radioactive bursts)
5. SPE-template convolution and FADC digitisation (3.3 ns bins)
"""

from collections import defaultdict

import numpy as np
import scipy.signal

# iDOM / module-level parameters
QE = 0.25           # total module quantum efficiency
TTS_NS = 2.0        # transit-time spread sigma [ns]
FADC_BIN_NS = 3.3
SIM_DT_NS = 0.1
PULSE_WIDTH_NS = 2.5
SPE_MEAN = 1.0
SPE_SIGMA = 0.3

# mDOM multi-PMT parameters 
N_PMTS = 24                # PMTs per mDOM
PMT_DARK_RATE_HZ = 750.0   # per-PMT rate in seawater (thermal + ⁴⁰K), [Hz]
                           # 18 000 Hz total / 24 PMTs 

# Shower / track development in water, for emission-point estimates
X0_WATER_M = 0.361         # radiation length [m]
EC_WATER_GEV = 0.0787      # EM critical energy [GeV]
LAMBDA_I_WATER_M = 0.83    # nuclear interaction length [m]
MUON_DEDX_GEV_PER_M = 0.2  # minimum-ionising dE/dx [GeV/m]
MUON_MASS_GEV = 0.10566


def emission_length(pdg: int, energy: float) -> float:
    """Return the distance from the vertex to the mean light-emission point [m].

    Photons reaching a module travel in straight lines from where they were
    emitted, so the arrival direction at the module is set by the emission
    point, not by the Cherenkov angle.  This helper places each particle's
    light at a characteristic depth: shower maximum for EM showers, the track
    midpoint for muons, and roughly one half interaction length for hadrons.

    Parameters
    ----------
    pdg : int
        Particle PDG code.
    energy : float
        Total particle energy [GeV].

    Returns
    -------
    float
        Emission distance along the particle direction [m].
    """
    a = abs(int(pdg))
    if a in (11, 22):
        return X0_WATER_M * max(np.log(max(energy, 1e-3) / EC_WATER_GEV), 0.5)
    if a == 13:
        e_kin = max(energy - MUON_MASS_GEV, 0.0)
        return 0.5 * e_kin / MUON_DEDX_GEV_PER_M
    return 0.5 * LAMBDA_I_WATER_M


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return (n, 3) unit vectors uniformly distributed on the sphere.

    Uses the Fibonacci / golden-angle spiral for even coverage of the sphere.
    The z-axis is taken as the string (vertical) direction.

    Parameters
    ----------
    n : int
        Number of directions.

    Returns
    -------
    np.ndarray
        Shape (n, 3) unit vectors.
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n, dtype=float)
    theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n)   # polar  [0, π]
    phi   = 2.0 * np.pi * i / golden                  # azimuthal
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


# Pre-computed PMT directions in the module frame (fixed, z-axis = string axis).
PMT_DIRS: np.ndarray = fibonacci_sphere(N_PMTS)   # shape (24, 3)


def dark_noise(t_min: float, t_max: float, rate_hz: float,
               correlated_frac: float = 0.2,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample dark-noise hit times in a window.

    Combines a uniform thermal component with correlated radioactive bursts
    (exponentially spaced secondaries).

    Parameters
    ----------
    t_min, t_max : float
        Window bounds [ns].
    rate_hz : float
        Total dark-noise rate [Hz].
    correlated_frac : float
        Fraction of the rate attributed to correlated bursts.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Sorted noise hit times [ns].
    """
    if rng is None:
        rng = np.random.default_rng()
    dt_s = max(0.0, t_max - t_min) * 1e-9

    n_th = rng.poisson(rate_hz * (1.0 - correlated_frac) * dt_s)
    thermal = rng.uniform(t_min, t_max, n_th)

    burst_rate = rate_hz * correlated_frac / 2.5
    burst_times: list[float] = []
    for _ in range(rng.poisson(burst_rate * dt_s)):
        t0 = float(rng.uniform(t_min, t_max))
        burst_times.append(t0)
        n_sec = rng.poisson(1.5)
        if n_sec > 0:
            burst_times.extend((t0 + rng.exponential(15.0, n_sec)).tolist())

    parts = [thermal]
    if burst_times:
        parts.append(np.asarray(burst_times))
    return np.sort(np.concatenate(parts)) if len(parts) > 1 or n_th > 0 else np.array([])


def generate_fadc_response(
    photon_times: np.ndarray,
    qe: float = 1.0,          # QE already applied upstream when qe=1
    tts_ns: float = TTS_NS,
    dark_rate_hz: float = PMT_DARK_RATE_HZ,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply QE, TTS, dark noise, and FADC digitisation to photon arrival times.

    Mirrors ``generate_fast_fadc_response`` from
    ``a reference implementation``.

    Parameters
    ----------
    photon_times : ndarray
        Photon arrival times at the module face [ns].
    qe : float
        Quantum efficiency (detection probability per photon).
    tts_ns : float
        Transit-time spread sigma [ns].
    dark_rate_hz : float
        Total PMT dark-noise rate [Hz].
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    fadc_t : ndarray
        FADC bin centre times [ns].
    fadc_q : ndarray
        Charge in each non-zero bin [PE].
    n_signal_pe : int
        Number of detected signal photoelectrons before dark noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    times = np.asarray(photon_times, dtype=float)

    # QE filter (only applied when qe < 1; for per-PMT calls qe=1 since
    # assignment already encoded the geometric + QE probability)
    detected = rng.random(len(times)) < qe
    signal_t = times[detected] + rng.normal(0.0, tts_ns, detected.sum())
    n_signal_pe = len(signal_t)

    if n_signal_pe > 0:
        win_min = float(np.floor(signal_t.min())) - 100.0
        win_max = float(np.ceil(signal_t.max())) + 100.0
    else:
        win_min, win_max = 0.0, 1000.0

    noise_t = dark_noise(win_min, win_max, dark_rate_hz, rng=rng)
    all_t = np.sort(np.concatenate([signal_t, noise_t])) if len(noise_t) else np.sort(signal_t)

    if len(all_t) == 0:
        return np.array([]), np.array([]), n_signal_pe

    charges = np.clip(rng.normal(SPE_MEAN, SPE_SIGMA, len(all_t)), 0.1, None)

    # Sparse clustering to avoid allocating dense arrays over large windows
    gaps = np.diff(all_t)
    split_idx = np.where(gaps > 50.0)[0] + 1
    t_clusters = np.split(all_t, split_idx)
    q_clusters = np.split(charges, split_idx)

    template_t = np.arange(-15.0, 15.0, SIM_DT_NS)
    spe_template = np.exp(-0.5 * (template_t / PULSE_WIDTH_NS) ** 2)
    spe_template /= spe_template.sum()

    fadc_t_parts: list[np.ndarray] = []
    fadc_q_parts: list[np.ndarray] = []

    for ct, cq in zip(t_clusters, q_clusters):
        if len(ct) == 0:
            continue
        c_min = np.floor((ct[0] - 20.0) / FADC_BIN_NS) * FADC_BIN_NS
        c_max = np.ceil((ct[-1] + 20.0) / FADC_BIN_NS) * FADC_BIN_NS

        hi_bins = np.arange(c_min, c_max + SIM_DT_NS, SIM_DT_NS)
        hi_sig, _ = np.histogram(ct, bins=hi_bins, weights=cq)
        analog = scipy.signal.fftconvolve(hi_sig, spe_template, mode="same")
        t_analog = hi_bins[:-1] + SIM_DT_NS / 2.0

        fadc_bins = np.arange(c_min, c_max + FADC_BIN_NS, FADC_BIN_NS)
        digi_q, _ = np.histogram(t_analog, bins=fadc_bins, weights=analog)
        fadc_t_cluster = fadc_bins[:-1] + FADC_BIN_NS / 2.0

        mask = digi_q > 0.05
        fadc_t_parts.append(fadc_t_cluster[mask])
        fadc_q_parts.append(digi_q[mask])

    if not fadc_t_parts:
        return np.array([]), np.array([]), n_signal_pe

    fadc_t = np.concatenate(fadc_t_parts)
    fadc_q = np.concatenate(fadc_q_parts)
    order = np.argsort(fadc_t)
    return fadc_t[order], fadc_q[order], n_signal_pe


def assign_to_pmts(
    photon_times: np.ndarray,
    source_dir: np.ndarray,
    pmt_dirs: np.ndarray,
    qe: float,
    rng: np.random.Generator,
) -> dict:
    """Assign photons to individual PMTs using Lambert's cosine-law visibility.

    Parameters
    ----------
    photon_times : ndarray
        Arrival times at the module centre [ns].
    source_dir : ndarray, shape (3,)
        Unit vector pointing FROM the module TOWARD the photon source (vertex).
        PMTs whose normals align with this direction are the "lit" ones.
    pmt_dirs : ndarray, shape (N_PMT, 3)
        Outward-facing unit normals of each PMT.
    qe : float
        Total module quantum efficiency.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Mapping of pmt_index -> array of detected photon times.
    """
    source_dirs = np.broadcast_to(source_dir, (len(photon_times), 3))
    return assign_to_pmts_per_hit(photon_times, source_dirs, pmt_dirs, qe, rng)


def assign_to_pmts_per_hit(
    photon_times: np.ndarray,
    source_dirs: np.ndarray,
    pmt_dirs: np.ndarray,
    qe: float,
    rng: np.random.Generator,
) -> dict:
    """Assign photons to PMTs with a per-photon arrival direction.

    Generalises :func:`assign_to_pmts`: each photon carries its own unit
    vector pointing from the module toward its emission point, so photons
    from different particles of the same event illuminate different PMT
    subsets.  This is what preserves event topology in the intra-module hit
    pattern.

    Parameters
    ----------
    photon_times : ndarray
        Arrival times at the module centre [ns].
    source_dirs : ndarray, shape (n_photons, 3)
        Per-photon unit vectors from the module toward the emission point.
    pmt_dirs : ndarray, shape (N_PMT, 3)
        Outward-facing unit normals of each PMT.
    qe : float
        Total module quantum efficiency.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Mapping of pmt_index -> array of detected photon times.
    """
    n_pmts = len(pmt_dirs)
    uniform = np.ones(n_pmts) / n_pmts

    pmt_times: dict[int, list] = defaultdict(list)
    for t, s_dir in zip(photon_times, source_dirs):
        if rng.random() >= qe:
            continue
        vis = np.maximum(0.0, pmt_dirs @ s_dir)
        vis_sum = vis.sum()
        p = vis / vis_sum if vis_sum > 0.0 else uniform
        pmt_idx = int(rng.choice(n_pmts, p=p))
        pmt_times[pmt_idx].append(float(t))
    return {k: np.array(v) for k, v in pmt_times.items()}


def process_event(
    photons: dict,
    vertex_pos: np.ndarray,
    rng: np.random.Generator,
    qe: float = QE,
    dark_rate_hz: float = PMT_DARK_RATE_HZ,
    source_points: np.ndarray = None,
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

    Returns
    -------
    dict
        One entry per *PMT* that fired, with keys ``string_id``, ``sensor_id``,
        ``sensor_pos_x/y/z``, ``pmt_id``, ``pmt_dir_x/y/z``, ``n_pe``,
        ``fadc_t``, ``fadc_q``.
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
            # Per-PMT FADC (qe=1 since assignment already applied QE)
            fadc_t, fadc_q, n_pe = generate_fadc_response(
                hit_times, qe=1.0, dark_rate_hz=dark_rate_hz, rng=rng
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
    return out
