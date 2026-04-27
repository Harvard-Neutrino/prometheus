"""Utility functions for event generation.
"""

import logging

import jax.numpy as jnp
import numpy as np

from prometheus.utils.geo_utils import is_in_cylinder, track_isects_cyl  # noqa: F401

from .constants import Constants

logger = logging.getLogger(__name__)


def sph_to_cart_jnp(theta, phi=0):
    """Transform spherical to cartesian coordinates.

    Parameters
    ----------
    theta : float
        Polar angle.
    phi : float, optional
        Azimuthal angle (default is 0).

    Returns
    -------
    jax.numpy.ndarray
        Cartesian coordinates (x, y, z) as a JAX array.
    """
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

    return jnp.asarray([x, y, z], dtype=jnp.float64)


def t_geo(x, t_0, direc, x_0):
    """Calculate the expected arrival time of unscattered photons at position ``x`` emitted by a muon.

    Parameters
    ----------
    x : np.ndarray
        Position of the sensor (shape (3,)).
    t_0 : float
        Time at which the muon is at ``x_0``.
    direc : np.ndarray
        Normalized direction vector of the muon.
    x_0 : np.ndarray
        Position of the muon at time ``t_0``.

    Returns
    -------
    float
        Expected arrival time of unscattered photons.
    """
    q = np.linalg.norm(np.cross((x - x_0), direc))
    return t_0 + 1 / Constants.c_vac * (
        np.dot(direc, (x - x_0))
        + q * (Constants.n_gr * Constants.n_ph - 1) / np.sqrt((Constants.n_ph**2) - 1)
    )


def proposal_setup():
    """Set up a PROPOSAL propagator.

    Returns
    -------
    object
        Configured PROPOSAL propagator.
    """
    try:
        import proposal as pp
    except ImportError as e:
        logger.critical("Could not import proposal!")
        raise e

    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Water(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(500, 1, False),
    }

    cross = pp.crosssection.make_std_crosssection(**args)  # use the standard crosssections
    collection = pp.PropagationUtilityCollection()

    collection.displacement = pp.make_displacement(cross, True)
    collection.interaction = pp.make_interaction(cross, True)
    collection.time = pp.make_time(cross, args["particle_def"], True)

    utility = pp.PropagationUtility(collection=collection)

    detector = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1e20)
    density_distr = pp.density_distribution.density_homogeneous(args["target"].mass_density)
    prop = pp.Propagator(args["particle_def"], [(detector, utility, density_distr)])
    return prop


def deposited_energy(det, record):
    """Calculate the deposited energy inside the detector outer hull.

    Parameters
    ----------
    det : object
        Detector instance providing ``outer_cylinder``.
    record : object
        Event record containing ``sources`` with amplitudes.

    Returns
    -------
    float
        Deposited energy inside the hull.
    """
    dep_e = 0
    for source in record.sources:
        if is_in_cylinder(det.outer_cylinder[0], det.outer_cylinder[1], source.pos):
            dep_e += source.amp
    return dep_e
