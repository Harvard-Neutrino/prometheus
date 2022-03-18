"""Light yield calculation."""
import jax
import jax.numpy as jnp
import numpy as np
from fennel import Fennel, config
from jax import random

from ..utils import rotate_to_new_direc_v
from .constants import Constants

config["general"]["jax"] = True
fennel_instance = Fennel()


def simple_cascade_light_yield(energy, *args):
    """
    Approximation for cascade light yield.

    Parameters:
        energy: float
            Particle energy in GeV
    """
    photons_per_GeV = 5.3 * 250 * 1e2

    return energy * photons_per_GeV


def fennel_total_light_yield(energy, particle_id, wavelength_range):
    """
    Calculate total light yield using fennel.

    Parameters:
        energy: float
            Particle energy in GeV
        particle_id: int
        wavelength_range: tuple
    """

    funcs = fennel_instance.auto_yields(energy, particle_id, function=True)
    counts_func = funcs[0]

    wavelengths = jnp.linspace(wavelength_range[0], wavelength_range[1], 100)
    light_yield = jnp.trapz(counts_func(energy, wavelengths).ravel(), wavelengths)

    return light_yield


def fennel_frac_long_light_yield(energy, particle_id, resolution=0.2):
    """
    Calculate the longitudinal light yield contribution.

    Integrate the longitudinal distribution in steps of `resolution` and
    return the relative contributions.

    Parameters:
        energy: float
            Particle energy in GeV
        particle_id: int
        resolution: float
            Step length in m for evaluating the longitudinal distribution
    """
    funcs = fennel_instance.auto_yields(energy, particle_id, function=True)
    long_func = funcs[4]
    int_grid = jnp.arange(1e-3, 25, resolution)

    def integrate(low, high, resolution=1000):
        trapz_x_eval = jnp.linspace(low, high, resolution) * 100  # to cm
        trapz_y_eval = long_func(energy, trapz_x_eval)
        return jnp.trapz(trapz_y_eval, trapz_x_eval)

    integrate_v = jax.vmap(integrate, in_axes=[0, 0])

    norm = integrate(1e-3, 100)
    frac_yields = integrate_v(int_grid[:-1], int_grid[1:]) / norm

    return frac_yields, int_grid


def make_pointlike_cascade_source(
    pos,
    t0,
    dir,
    energy,
    particle_id,
    wavelength_range=[290, 700],
):
    """
    Create a pointlike lightsource.

    Parameters:
        pos: float[3]
            Cascade position
        t0: float
            Cascade time
        dir: float[3]
            Cascade direction
        energy: float
            Cascade energy
        particle_id: int
            Particle type (PDG ID)
        wavelength_range: tuple
            Wavelength interval (nm)


    Returns:
        List[PhotonSource]

    """
    source_nphotons = jnp.asarray(
        [fennel_total_light_yield(energy, particle_id, wavelength_range)]
    )[np.newaxis, :]

    source_pos = pos[np.newaxis, :]
    source_dir = dir[np.newaxis, :]
    source_time = jnp.asarray([t0])[np.newaxis, :]
    # source = PhotonSource(pos, n_photons, t0, dir)
    return source_pos, source_dir, source_time, source_nphotons


rotate_to_new_direc_v


def make_realistic_cascade_source(
    pos,
    t0,
    dir,
    energy,
    particle_id,
    key,
    resolution=0.2,
    moliere_rand=False,
    wavelength_range=[290, 700],
):
    """
    Create a realistic (elongated) particle cascade.

    The longitudinal profile is approximated by placing point-like light sources
    every `resolution` steps.

    Parameters:
        pos: float[3]
            Cascade position
        t0: float
            Cascade time
        dir: float[3]
            Cascade direction
        energy: float
            Cascade energy
        particle_id: int
            Particle type (PDG ID)
        key: PRNGKey
            Random key
        resolution: float
            Step size for point-like light sources
        moliere_rand: bool
            Switch moliere randomization
        wavelength_range: tuple
            Wavelength interval (nm)
    """
    n_photons_total = fennel_total_light_yield(energy, particle_id, wavelength_range)
    frac_yields, grid = fennel_frac_long_light_yield(energy, particle_id, resolution)

    dist_along = 0.5 * (grid[:-1] + grid[1:])

    if moliere_rand:
        moliere_radius = 0.21
        key, k1, k2 = random.split(key, 3)
        r = moliere_radius * jnp.sqrt(random.uniform(k1, shape=dist_along.shape))
        phi = random.uniform(k2, shape=dist_along.shape, maxval=2 * np.pi)
        x = r * jnp.cos(phi)
        y = r * jnp.sin(phi)

        dpos_vec = jnp.stack([x, y, jnp.zeros_like(x)], axis=1)
        dpos_vec = rotate_to_new_direc_v(jnp.asarray([0, 0, 1]), dir, dpos_vec)
        source_pos = (
            dist_along[:, np.newaxis] * dir[np.newaxis, :]
            + pos[np.newaxis, :]
            + dpos_vec
        )
    else:
        source_pos = dist_along[:, np.newaxis] * dir[np.newaxis, :] + pos[np.newaxis, :]

    source_dir = jnp.tile(dir, (dist_along.shape[0], 1))
    source_nphotons = frac_yields * n_photons_total
    source_time = t0 + dist_along / (Constants.c_vac)

    return (
        source_pos,
        source_dir,
        source_time[:, np.newaxis],
        source_nphotons[:, np.newaxis],
    )
