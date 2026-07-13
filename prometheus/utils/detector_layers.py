# -*- coding: utf-8 -*-
# detector_layers.py
# Radial layering / Voronoi-cell geometry for layered detector simulations.

"""Detector geometry for the radial-layering approximation.

String-based detectors are approximately translation symmetric: strings at
similar radii see similar photon yields.  These helpers partition the strings
into radial layers and pick one representative string per layer, together
with the Voronoi cell geometry used to sample event vertices around it (see
:mod:`prometheus.utils.layered_sim` for the vertex sampling and simulation
execution built on top of this geometry).
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


def representative_string(
    strings: np.ndarray, radii: np.ndarray, indices: np.ndarray
) -> np.ndarray:
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
        layers.append(
            {
                "n_strings": len(indices),
                "r_range": (float(radii[indices].min()), float(radii[indices].max())),
                "rep_xy": representative_string(strings, radii, indices),
            }
        )
    return layers, cell


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
        "R_det_m": cell.R_det,
        "d_nn_m": cell.d_nn,
        "V_cell_m3": cell.V_cell,
        "r_cell_m": cell.r_cell,
        "z_min_m": cell.z_min,
        "z_max_m": cell.z_max,
        "H_m": cell.H,
    }
