"""Detector helpers and re-exports.

This module re-exports canonical detector classes from ``prometheus.detector``
and provides sampling utilities used by event generators.
"""

# Re-export shared symbols from the canonical prometheus.detector package.

import awkward as ak
import numpy as np


def sample_cylinder_surface(height, radius, n, rng=np.random.RandomState(1337)):
    """Sample points on a cylinder surface.

    Parameters
    ----------
    height : float
        Cylinder height.
    radius : float
        Cylinder radius.
    n : int
        Number of samples to draw.
    rng : np.random.RandomState, optional
        Random number generator (default seeded RandomState(1337)).

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) with sampled Cartesian coordinates on the cylinder surface.
    """
    side_area = 2 * np.pi * radius * height
    top_area = 2 * np.pi * radius**2

    ratio = top_area / (top_area + side_area)

    is_top = rng.uniform(0, 1, size=n) < ratio
    n_is_top = is_top.sum()
    samples = np.empty((n, 3))
    theta = rng.uniform(0, 2 * np.pi, size=n)

    # top / bottom points

    r = radius * np.sqrt(rng.uniform(0, 1, size=n_is_top))

    samples[is_top, 0] = r * np.sin(theta[is_top])
    samples[is_top, 1] = r * np.cos(theta[is_top])
    samples[is_top, 2] = rng.choice([-height / 2, height / 2], replace=True, size=n_is_top)

    # side points

    r = radius
    samples[~is_top, 0] = r * np.sin(theta[~is_top])
    samples[~is_top, 1] = r * np.cos(theta[~is_top])
    samples[~is_top, 2] = rng.uniform(-height / 2, height / 2, size=n - n_is_top)

    return samples


def sample_cylinder_volume(height, radius, n, rng=np.random.RandomState(1337)):
    """Sample points uniformly inside a cylinder volume.

    Parameters
    ----------
    height : float
        Cylinder height.
    radius : float
        Cylinder radius.
    n : int
        Number of samples.
    rng : np.random.RandomState, optional
        Random number generator (default seeded RandomState(1337)).

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) with sampled Cartesian coordinates inside the cylinder.
    """
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = radius * np.sqrt(rng.uniform(0, 1, size=n))
    samples = np.empty((n, 3))
    samples[:, 0] = r * np.sin(theta)
    samples[:, 1] = r * np.cos(theta)
    samples[:, 2] = rng.uniform(-height / 2, height / 2, size=n)
    return samples


def sample_direction(n_samples, rng=np.random.RandomState(1337)):
    """Sample uniform unit directions on the sphere.

    Parameters
    ----------
    n_samples : int
        Number of direction samples to draw.
    rng : np.random.RandomState, optional
        Random number generator (default seeded RandomState(1337)).

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 3) with unit direction vectors.
    """
    cos_theta = rng.uniform(-1, 1, size=n_samples)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0, 2 * np.pi)

    samples = np.empty((n_samples, 3))
    samples[:, 0] = np.sin(theta) * np.cos(phi)
    samples[:, 1] = np.sin(theta) * np.sin(phi)
    samples[:, 2] = np.cos(theta)

    return samples


def get_proj_area_for_zen(height, radius, coszen):
    """Return projected area of a cylinder for a given zenith cosine.

    Parameters
    ----------
    height : float
        Cylinder height.
    radius : float
        Cylinder radius.
    coszen : float
        Cosine of the zenith angle.

    Returns
    -------
    float
        Projected area.
    """
    cap = np.pi * radius * radius
    sides = 2 * radius * height
    return cap * np.abs(coszen) + sides * np.sqrt(1.0 - coszen * coszen)


def generate_noise(det, time_range, rng=np.random.RandomState(1337)):
    """Generate detector noise within a time range for each module.

    Parameters
    ----------
    det : Detector
        Detector instance providing `modules` with `noise_rate`.
    time_range : sequence
        Two-element sequence specifying start and end time.
    rng : np.random.RandomState, optional
        Random number generator (default seeded RandomState(1337)).

    Returns
    -------
    ak.Array
        Sorted per-module noise hit times.
    """
    all_times_det = []
    dT = np.diff(time_range)
    for idom in range(len(det.modules)):
        noise_amp = rng.poisson(det.modules[idom].noise_rate * dT)
        times_det = rng.uniform(*time_range, size=noise_amp)
        all_times_det.append(times_det)

    return ak.sort(ak.Array(all_times_det))


def trigger(det, event_times, mod_thresh=8, phot_thres=5):
    """Check a simple multiplicity trigger condition.

    Trigger is true when at least ``mod_thresh`` modules have measured more than
    ``phot_thres`` photons.

    Parameters
    ----------
    det : Detector
        Detector instance.
    event_times : ak.Array
        Per-module photon arrival times.
    mod_thresh : int, optional
        Threshold for the number of modules which have detected ``phot_thres`` photons.
    phot_thres : int, optional
        Threshold for the number of photons per module.

    Returns
    -------
    bool
        ``True`` when the multiplicity condition is met, otherwise ``False``.
    """
    hits_per_module = ak.count(event_times, axis=1)
    if ak.sum((hits_per_module > phot_thres)) > mod_thresh:
        return True
    return False


# def local_coinc(hit_times, lc_links, pmt_t=50, lc_t=500, smt_t=1000):

#     trigger_times = []
#     mod_ids = []
#     lc_c
#     for mid in range(len(hit_times)):
#         ts_l = ak.sort(ak.flatten(hit_times[lc_links[mid]]))
#         ts_mod = hit_times[mid]

#         # More than two hits within 50 ns
#         valid = (ts_mod[1:] - ts_mod[:-1]) < pmt_t

#         triggers = np.zeros(ak.sum(valid), dtype=np.bool)
#         for i, vhit in enumerate(ts_mod[valid]):

#             # At least one hit within 500ns on neighboring module
#             if np.any(np.abs(ts_l - vhit) < lc_t):
#                 triggers[i] = True
#         trigger_times.append(ts_mod[valid][triggers])
#         mod_ids.append(np.ones(triggers.shape[0]) * mid)

#     trigger_times = ak.concatenate(trigger_times)
#     return ak.sum((trigger_times[1:] - trigger_times[:-1]) < smt_t)
