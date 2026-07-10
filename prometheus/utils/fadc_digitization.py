# -*- coding: utf-8 -*-
# fadc_digitization.py
# Dark noise, discriminator threshold-crossing, and FADC digitisation for the
# mDOM response model (split out of dom_response.py).

"""Analog pulse formation and digitisation for the mDOM response model.

1. Dark noise injection (thermal + correlated radioactive bursts)
   (:func:`dark_noise`).
2. SPE-template convolution and FADC digitisation, plus a discriminator
   threshold-crossing pass producing ToT hits -- the only readout a real
   KM3NeT DOM front-end provides (:func:`generate_fadc_response`,
   :func:`_threshold_crossings`).
"""

import numpy as np
import scipy.signal

from prometheus.config_types import DOMResponseConfig

_cfg = DOMResponseConfig()

TTS_NS = _cfg.tts_ns              # transit-time spread sigma [ns]
FADC_BIN_NS = _cfg.fadc_bin_ns
SIM_DT_NS = _cfg.sim_dt_ns
PULSE_WIDTH_NS = _cfg.pulse_width_ns
SPE_MEAN = _cfg.spe_mean
SPE_SIGMA = _cfg.spe_sigma

# Front-end discriminator / ToT digitisation (real KM3NeT DOMs report only a
# leading-edge hit time and a Time-over-Threshold duration, not a charge).
TOT_THRESHOLD_PE = _cfg.tot_threshold_pe  # discriminator threshold, PE-equivalent amplitude
TOT_MAX_NS = _cfg.tot_max_ns              # 8-bit TDC saturation cap

PMT_DARK_RATE_HZ = _cfg.pmt_dark_rate_hz  # per-PMT rate in seawater (thermal + ⁴⁰K), [Hz]
                                           # 18 000 Hz total / 24 PMTs 


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


def _threshold_crossings(
    amplitude: np.ndarray,
    t_analog: np.ndarray,
    threshold_pe: float,
    max_tot_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leading-edge times and ToT durations from an amplitude trace.

    Mirrors a real front-end discriminator: a hit is any contiguous run of
    samples above ``threshold_pe``; its ToT is the run's duration, floored to
    whole nanoseconds (a discriminator/TDC cannot report sub-ns durations)
    with a 1 ns floor, and capped at ``max_tot_ns`` (TDC saturation).
    Overlapping pulses (pile-up) merge into one longer run automatically
    since they are already summed in ``amplitude``.

    Parameters
    ----------
    amplitude : np.ndarray
        Instantaneous pulse amplitude, PE-equivalent (peak = 1 for an
        isolated single-PE hit), sampled on ``t_analog``.
    t_analog : np.ndarray
        Sample times [ns] of ``amplitude``.
    threshold_pe : float
        Discriminator threshold, PE-equivalent amplitude.
    max_tot_ns : float
        ToT saturation cap [ns].

    Returns
    -------
    hit_t : np.ndarray
        Leading-edge time of each hit [ns].
    tot_ns : np.ndarray
        ToT duration of each hit [ns].
    """
    above = amplitude > threshold_pe
    if not np.any(above):
        return np.array([]), np.array([])

    edges = np.diff(above.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]

    dt = t_analog[1] - t_analog[0]
    hit_t = t_analog[starts]
    duration = t_analog[ends - 1] - hit_t + dt
    tot_ns = np.minimum(np.maximum(np.floor(duration), 1.0), max_tot_ns)
    return hit_t, tot_ns


def generate_fadc_response(
    photon_times: np.ndarray,
    qe: float = 1.0,          # QE already applied upstream when qe=1
    tts_ns: float = TTS_NS,
    dark_rate_hz: float = PMT_DARK_RATE_HZ,
    rng: np.random.Generator | None = None,
    tot_threshold_pe: float = TOT_THRESHOLD_PE,
    tot_max_ns: float = TOT_MAX_NS,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Apply QE, TTS, dark noise, FADC digitisation, and ToT hit extraction.

    Mirrors ``generate_fast_fadc_response`` from
    ``a reference implementation``, plus
    a discriminator threshold-crossing pass over the same analog trace to
    produce ToT hits (leading-edge time, Time-over-Threshold duration) — the
    only readout a real KM3NeT DOM front-end provides.

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
    tot_threshold_pe : float
        Discriminator threshold, PE-equivalent amplitude.
    tot_max_ns : float
        ToT saturation cap [ns].

    Returns
    -------
    fadc_t : ndarray
        FADC bin centre times [ns].
    fadc_q : ndarray
        Charge in each non-zero bin [PE].
    n_signal_pe : int
        Number of detected signal photoelectrons before dark noise.
    hit_t : ndarray
        Leading-edge time of each ToT hit [ns].
    tot_ns : ndarray
        ToT duration of each hit [ns].
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
        return np.array([]), np.array([]), n_signal_pe, np.array([]), np.array([])

    charges = np.clip(rng.normal(SPE_MEAN, SPE_SIGMA, len(all_t)), 0.1, None)

    # Sparse clustering to avoid allocating dense arrays over large windows
    gaps = np.diff(all_t)
    split_idx = np.where(gaps > 50.0)[0] + 1
    t_clusters = np.split(all_t, split_idx)
    q_clusters = np.split(charges, split_idx)

    template_t = np.arange(-15.0, 15.0, SIM_DT_NS)
    spe_template_raw = np.exp(-0.5 * (template_t / PULSE_WIDTH_NS) ** 2)
    template_peak_scale = spe_template_raw.sum()
    spe_template = spe_template_raw / template_peak_scale

    fadc_t_parts: list[np.ndarray] = []
    fadc_q_parts: list[np.ndarray] = []
    hit_t_parts: list[np.ndarray] = []
    tot_ns_parts: list[np.ndarray] = []

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

        # ToT: threshold-crossing on the (un-normalised) amplitude trace, so
        # an isolated single-PE hit peaks at ~1.0 PE-equivalent amplitude.
        amplitude = analog * template_peak_scale
        hit_t_cluster, tot_ns_cluster = _threshold_crossings(
            amplitude, t_analog, tot_threshold_pe, tot_max_ns
        )
        hit_t_parts.append(hit_t_cluster)
        tot_ns_parts.append(tot_ns_cluster)

    if not fadc_t_parts:
        return np.array([]), np.array([]), n_signal_pe, np.array([]), np.array([])

    fadc_t = np.concatenate(fadc_t_parts)
    fadc_q = np.concatenate(fadc_q_parts)
    order = np.argsort(fadc_t)

    hit_t = np.concatenate(hit_t_parts)
    tot_ns = np.concatenate(tot_ns_parts)
    tot_order = np.argsort(hit_t)

    return (fadc_t[order], fadc_q[order], n_signal_pe,
            hit_t[tot_order], tot_ns[tot_order])
