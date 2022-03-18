"""Utility functions."""
import logging

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from .constants import Constants

logger = logging.getLogger(__name__)


def sph_to_cart_jnp(theta, phi=0):
    """Transform spherical to cartesian coordinates."""
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

    return jnp.asarray([x, y, z], dtype=jnp.float64)


def t_geo(x, t_0, direc, x_0):
    """
    Calculate the expected arrival time of unscattered photons.

    Calculate the expected arrival time of unscattered photons. at position `x`,
    emitted by a muon with direction `direc` and time `t_0` at position `x_0`.

    Parameters:
      x: (3,1) np.ndarray
        position of the sensor
      t_0: float
        time at which muon is at `x_0`
      direc: (3,1) np.ndarray
        normalized direction vector of the muon
      x_0: (3, 1) np.ndarray
    """
    q = np.linalg.norm(np.cross((x - x_0), direc))
    return t_0 + 1 / Constants.c_vac * (
        np.dot(direc, (x - x_0))
        + q * (Constants.n_gr * Constants.n_ph - 1) / np.sqrt((Constants.n_ph ** 2) - 1)
    )


def proposal_setup():
    """Set up a proposal propagator."""
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

    cross = pp.crosssection.make_std_crosssection(
        **args
    )  # use the standard crosssections
    collection = pp.PropagationUtilityCollection()

    collection.displacement = pp.make_displacement(cross, True)
    collection.interaction = pp.make_interaction(cross, True)
    collection.time = pp.make_time(cross, args["particle_def"], True)

    utility = pp.PropagationUtility(collection=collection)

    detector = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1e20)
    density_distr = pp.density_distribution.density_homogeneous(
        args["target"].mass_density
    )
    prop = pp.Propagator(args["particle_def"], [(detector, utility, density_distr)])
    return prop


def is_in_cylinder(radius, height, pos):
    """Test whether a position vector is inside a cylinder."""
    return (np.sqrt(pos[0] ** 2 + pos[1] ** 2) < radius) & (np.abs(pos[2]) < height / 2)


def get_zen_azi(direc):
    """Convert a cartesian direction into zenith / azimuth (IC convention)."""
    r = np.linalg.norm(direc)
    theta = 0
    if direc[2] / r <= 1:
        theta = np.arccos(direc[2] / r)
    else:
        if direc[2] < 0:
            theta = np.pi
    if theta < 0:
        theta += 2 * np.pi
    phi = 0
    if (direc[0] != 0) or (direc[1] != 0):
        phi = np.arctan2(direc[1], direc[0])
    if phi < 0:
        phi += 2 * np.pi
    zenith = np.pi - theta
    azimuth = phi + np.pi
    if zenith > np.pi:
        zenith -= 2 * np.pi - zenith
    azimuth -= int(azimuth / (2 * np.pi)) * (2 * np.pi)
    return zenith, azimuth


def track_isects_cyl(radius, height, pos, direc):
    """Check if a track intersects a cylinder."""
    x = pos[0]
    y = pos[1]
    z = pos[2]

    theta, phi = get_zen_azi(direc)

    sinph = np.sin(phi)
    cosph = np.cos(phi)
    sinth = np.sin(theta)
    costh = np.cos(theta)

    b = x * cosph + y * sinph
    d = b * b + radius * radius - x * x - y * y
    h = (np.nan, np.nan)
    r = (np.nan, np.nan)

    if d > 0:
        d = np.sqrt(d)
        # down-track distance to the endcaps
        if costh != 0:
            h = sorted(((z - height / 2) / costh, (z + height / 2) / costh))
        # down-track distance to the side surfaces
        if sinth != 0:
            r = sorted(((b - d) / sinth, (b + d) / sinth))

        if costh == 0:
            if (z > -height / 2) & (z < height / 2):
                h = r
            else:
                h = (np.nan, np.nan)
        elif sinth == 0:
            if np.sqrt(x ** 2 + y ** 2) >= radius:
                h = (np.nan, np.nan)
        else:

            if (h[0] >= r[1]) or (h[1] <= r[0]):
                h = (np.nan, np.nan)
            else:
                h = max(h[0], r[0]), min(h[1], r[1])
    return h


def deposited_energy(det, record):
    """Calculate the deposited energy inside the detector outer hull."""
    dep_e = 0
    for source in record.sources:
        if is_in_cylinder(det.outer_cylinder[0], det.outer_cylinder[1], source.pos):
            dep_e += source.amp
    return dep_e
