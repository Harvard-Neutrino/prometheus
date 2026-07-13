# -*- coding: utf-8 -*-
# layered_sim.py
# Layered detector simulation execution shared by the effective-volume,
# nucleon-decay signal, and atmospheric-background examples.

"""Layered detector simulation execution.

Builds on the radial-layer/Voronoi-cell geometry in
:mod:`prometheus.utils.detector_layers` (re-exported here so existing
``from prometheus.utils.layered_sim import X`` call sites keep working):
sample event vertices inside a representative layer's cell, run one batch of
Prometheus events, and gather per-event hit statistics.  Scaling the
per-layer result by the layer's string count then estimates full-detector
quantities at a fraction of the simulation cost.
"""

import numpy as np

from prometheus.utils.detector_layers import (  # noqa: F401
    CellGeometry,
    build_layers,
    cell_geometry,
    cell_manifest_dict,
    extract_strings,
    partition_layers,
    representative_string,
)


def sample_cell_vertices(
    rng: np.random.Generator,
    string_xy: np.ndarray,
    r_cell: float,
    z_min: float,
    z_max: float,
    n: int,
) -> np.ndarray:
    """Sample vertex positions uniformly inside a cylindrical Voronoi cell.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    string_xy : np.ndarray
        Shape (2,) centre of the cell in the horizontal plane.
    r_cell : float
        Cell radius [m].
    z_min, z_max : float
        Vertical extent of the detector [m].
    n : int
        Number of vertices to sample.

    Returns
    -------
    np.ndarray
        Shape (n, 3) vertex positions.
    """
    r = r_cell * np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    x = string_xy[0] + r * np.cos(theta)
    y = string_xy[1] + r * np.sin(theta)
    z = rng.uniform(z_min, z_max, n)
    return np.column_stack([x, y, z])


def wilson_sigma(k: int, n: int, z: float = 1.0) -> float:
    """Return the Wilson-interval half-width for ``k`` successes in ``n`` trials.

    Unlike the Wald error ``sqrt(p(1-p)/n)``, this stays finite (and honest)
    when the observed efficiency is exactly 0 or 1.

    Parameters
    ----------
    k : int
        Number of detected events.
    n : int
        Number of simulated events.
    z : float
        Number of standard deviations (z = 1 for a 68 % interval).

    Returns
    -------
    float
        Half-width of the Wilson score interval.
    """
    if n == 0:
        return 0.0
    return z * np.sqrt(k * (n - k) / n + z**2 / 4.0) / (n + z**2)


def count_hits_particle(particle) -> tuple:
    """Return (total_hits, set_of_hit_module_keys) for a particle and its children.

    Parameters
    ----------
    particle : PropagatableParticle
        Particle whose hits (and children's hits) are counted.

    Returns
    -------
    tuple of (int, set)
        Total raw photon hit count and the set of (string_id, om_id) pairs hit.
    """
    hits = getattr(particle, "hits", None) or []
    n_hits = len(hits)
    modules = {(h.string_id, h.om_id) for h in hits}

    for child in getattr(particle, "children", None) or []:
        c_hits, c_mods = count_hits_particle(child)
        n_hits += c_hits
        modules |= c_mods

    return n_hits, modules


def event_stats(injection) -> tuple:
    """Return per-event total hit counts and distinct module counts.

    Parameters
    ----------
    injection : Injection
        Injection whose events carry propagated hits.

    Returns
    -------
    hit_counts : np.ndarray
        Total raw photon hits per event.
    module_counts : np.ndarray
        Number of distinct modules hit per event.
    """
    hit_counts = []
    module_counts = []
    for event in injection:
        total_hits = 0
        all_modules = set()
        for fs in event.final_states:
            h, m = count_hits_particle(fs)
            total_hits += h
            all_modules |= m
        hit_counts.append(total_hits)
        module_counts.append(len(all_modules))
    return np.array(hit_counts), np.array(module_counts)


def run_batch(prom, config, rng, rep_xy, n_events, cell, seed, outfile):
    """Simulate one batch of events around a representative string.

    Parameters
    ----------
    prom : Prometheus
        Initialised Prometheus instance.
    config : PrometheusConfig
        Global Prometheus configuration object.
    rng : np.random.Generator
        Random number generator for vertex placement.
    rep_xy : np.ndarray
        Shape (2,) xy position of the representative string.
    n_events : int
        Number of events to simulate.
    cell : CellGeometry
        Cell geometry from :func:`prometheus.utils.detector_layers.cell_geometry`
        or :func:`prometheus.utils.detector_layers.build_layers`.
    seed : int
        Run seed for this batch; controls both the GENIE event resampling and
        the photon sampling.
    outfile : str or Path
        Parquet output file for this batch.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Per-event total hit counts and distinct module counts.
    """
    positions = sample_cell_vertices(rng, rep_xy, cell.r_cell, cell.z_min, cell.z_max, n_events)
    config.injection.genie.simulation.positions = positions.tolist()
    config.injection.genie.simulation.n_events = n_events
    # inject() copies the run seed into the injection config, and propagate()
    # derives its PRNG key from it, so a per-batch run seed decorrelates both.
    config.run.random_state_seed = seed
    config.run.outfile = str(outfile)

    # PROPOSAL's global generator is otherwise seeded only once at Prometheus
    # construction; re-seeding per batch makes muon energy losses reproducible
    # independent of how many batches ran before this one.
    import proposal as pp

    pp.RandomGenerator.get().set_seed(seed)

    prom.inject()
    prom.propagate()
    prom.construct_output()
    return event_stats(prom.injection)
