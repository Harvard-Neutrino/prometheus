# -*- coding: utf-8 -*-
# pmt_response.py
# Per-PMT photon assignment and emission-point geometry for the mDOM response
# model (split out of dom_response.py).

"""Per-PMT visibility and emission-point geometry for the mDOM response model.

1. Places each particle's light at a characteristic emission point
   (:func:`emission_length`).
2. Distributes photons across a module's PMTs using Lambert cosine-law
   visibility (:func:`assign_to_pmts`, :func:`assign_to_pmts_per_hit`).

PMT directions are laid out on a Fibonacci/golden-angle spiral
(:func:`fibonacci_sphere`), fixed once as :data:`PMT_DIRS`.
"""

from collections import defaultdict

import numpy as np

from prometheus.config_types import DOMResponseConfig

_cfg = DOMResponseConfig()

N_PMTS = _cfg.n_pmts

# Shower / track development in water, for emission-point estimates
X0_WATER_M = _cfg.x0_water_m             # radiation length [m]
EC_WATER_GEV = _cfg.ec_water_gev          # EM critical energy [GeV]
LAMBDA_I_WATER_M = _cfg.lambda_i_water_m  # nuclear interaction length [m]
MUON_DEDX_GEV_PER_M = _cfg.muon_dedx_gev_per_m  # minimum-ionising dE/dx [GeV/m]
MUON_MASS_GEV = _cfg.muon_mass_gev


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

    Uses the Fibonacci / golden-angle spiral for even coverage of the
    sphere. The z-axis is taken as the string (vertical) direction.

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
