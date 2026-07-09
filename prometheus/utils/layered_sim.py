# -*- coding: utf-8 -*-
# layered_sim.py
# Helpers for layered detector simulations shared by the effective-volume,
# nucleon-decay signal, and atmospheric-background examples.

"""Helpers for layered detector simulations.

String-based detectors are approximately translation symmetric: strings at
similar radii see similar photon yields.  These helpers partition the strings
into radial layers, pick one representative string per layer, and sample event
vertices inside a cylindrical Voronoi cell around it.  Scaling the per-layer
result by the layer's string count then estimates full-detector quantities at
a fraction of the simulation cost.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CellGeometry:
    """Voronoi cell geometry derived from a detector's string layout.

    Attributes
    ----------
    n_strings : int
        Number of strings in the detector.
    R_outer : float
        Radius of the outermost string [m].
    d_nn : float
        Mean nearest-neighbour string spacing [m].
    R_det : float
        Effective detector footprint radius, ``R_outer + d_nn / 2`` [m].
        The half-spacing skirt accounts for the detector volume beyond the
        outermost string ring.
    V_cell : float
        Mean Voronoi cell volume, ``pi R_det^2 H / n_strings`` [m^3].
    r_cell : float
        Effective cell radius, ``sqrt(R_det^2 / n_strings)`` [m].
    z_min, z_max : float
        Vertical extent of the instrumented volume [m].
    H : float
        Detector height, ``z_max - z_min`` [m].
    """

    n_strings: int
    R_outer: float
    d_nn: float
    R_det: float
    V_cell: float
    r_cell: float
    z_min: float
    z_max: float
    H: float


def extract_strings(detector) -> np.ndarray:
    """Return string (x, y) positions from the detector module keys.

    Grouping by string id (rather than de-duplicating rounded xy positions)
    stays correct for tilted strings or per-module position scatter.

    Parameters
    ----------
    detector : Detector
        Prometheus detector object.

    Returns
    -------
    np.ndarray
        (N_strings, 2) array of string xy positions (mean over each string's
        modules).
    """
    string_ids = np.asarray([m.key[0] for m in detector.modules])
    xy = detector.module_coords[:, :2]
    return np.array([xy[string_ids == sid].mean(axis=0) for sid in np.unique(string_ids)])


def cell_geometry(detector, strings: np.ndarray) -> CellGeometry:
    """Build the Voronoi cell geometry for a detector's string layout.

    Parameters
    ----------
    detector : Detector
        Prometheus detector object.
    strings : np.ndarray
        (N_strings, 2) string xy positions, e.g. from :func:`extract_strings`.

    Returns
    -------
    CellGeometry
        Cell geometry summary used for vertex sampling and volume weighting.
    """
    n_strings = len(strings)
    coords = detector.module_coords

    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    H = z_max - z_min

    radii = np.linalg.norm(strings - detector.offset[:2], axis=1)
    R_outer = radii.max()

    # The detector footprint extends about half a string spacing beyond the
    # outermost string ring; extend R_outer by half the mean nearest-neighbour
    # string distance so the total cell volume covers that skirt.
    pair_dists = np.linalg.norm(strings[:, np.newaxis] - strings[np.newaxis, :], axis=-1)
    np.fill_diagonal(pair_dists, np.inf)
    d_nn = pair_dists.min(axis=1).mean()
    R_det = R_outer + 0.5 * d_nn

    return CellGeometry(
        n_strings=n_strings,
        R_outer=float(R_outer),
        d_nn=float(d_nn),
        R_det=float(R_det),
        V_cell=float(np.pi * R_det**2 * H / n_strings),
        r_cell=float(np.sqrt(R_det**2 / n_strings)),
        z_min=float(z_min),
        z_max=float(z_max),
        H=float(H),
    )


def partition_layers(radii: np.ndarray, n_layers: int) -> list:
    """Partition string indices into radial layers with equal string counts.

    Parameters
    ----------
    radii : np.ndarray
        Radial distance of each string from the detector centre.
    n_layers : int
        Number of layers to create.

    Returns
    -------
    list of np.ndarray
        Each element is an index array of strings belonging to that layer.
    """
    order = np.argsort(radii)
    return np.array_split(order, n_layers)


def representative_string(strings: np.ndarray, radii: np.ndarray,
                          indices: np.ndarray) -> np.ndarray:
    """Return the string position closest to the median radius in this layer.

    Parameters
    ----------
    strings : np.ndarray
        (N_strings, 2) array of all string xy positions.
    radii : np.ndarray
        Radial distances corresponding to ``strings``.
    indices : np.ndarray
        Indices of strings that belong to this layer.

    Returns
    -------
    np.ndarray
        Shape (2,) xy position of the representative string.
    """
    layer_radii = radii[indices]
    best = indices[np.argmin(np.abs(layer_radii - np.median(layer_radii)))]
    return strings[best]


def build_layers(detector, n_layers: int):
    """Build the radial layer description for a detector.

    Parameters
    ----------
    detector : Detector
        Prometheus detector object.
    n_layers : int
        Number of radial layers to partition strings into.

    Returns
    -------
    layers : list of dict
        One entry per layer with keys ``n_strings``, ``r_range`` (min/max
        string radius) and ``rep_xy`` (representative string position).
    cell : CellGeometry
        Cell geometry shared by all layers.
    """
    strings = extract_strings(detector)
    cell = cell_geometry(detector, strings)
    radii = np.linalg.norm(strings - detector.offset[:2], axis=1)

    layers = []
    for indices in partition_layers(radii, n_layers):
        layers.append({
            "n_strings": len(indices),
            "r_range": (float(radii[indices].min()), float(radii[indices].max())),
            "rep_xy": representative_string(strings, radii, indices),
        })
    return layers, cell


def sample_cell_vertices(rng: np.random.Generator,
                         string_xy: np.ndarray,
                         r_cell: float,
                         z_min: float,
                         z_max: float,
                         n: int) -> np.ndarray:
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


def cell_manifest_dict(cell: CellGeometry) -> dict:
    """Return the cell geometry as a dict with unit-suffixed manifest keys.

    Parameters
    ----------
    cell : CellGeometry
        Cell geometry to serialise.

    Returns
    -------
    dict
        Keys as written to the run manifests of the signal and background
        example scripts (e.g. ``V_cell_m3``, ``r_cell_m``).
    """
    return {
        "n_strings": cell.n_strings,
        "R_outer_m": cell.R_outer,
        "R_det_m":   cell.R_det,
        "d_nn_m":    cell.d_nn,
        "V_cell_m3": cell.V_cell,
        "r_cell_m":  cell.r_cell,
        "z_min_m":   cell.z_min,
        "z_max_m":   cell.z_max,
        "H_m":       cell.H,
    }


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

    for child in (getattr(particle, "children", None) or []):
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
        Cell geometry from :func:`cell_geometry` or :func:`build_layers`.
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
    positions = sample_cell_vertices(
        rng, rep_xy, cell.r_cell, cell.z_min, cell.z_max, n_events
    )
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
