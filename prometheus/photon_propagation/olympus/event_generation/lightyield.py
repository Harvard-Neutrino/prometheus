"""Light-yield calculations and source factories."""

import jax
import jax.numpy as jnp
import numpy as np
from fennel import Fennel, config
from jax import random

from ..utils import rotate_to_new_direc_v
from .constants import Constants

config["general"]["jax"] = True
fennel_instance = Fennel()
try:
    jax_trapz = jax.numpy.trapz
except AttributeError:
    jax_trapz = jax.scipy.integrate.trapezoid

# ---------------------------------------------------------------------------
# PDG-code normalisation for particles not natively supported by fennel.
# ---------------------------------------------------------------------------

# GENIE pseudo-particles and nuclear remnants: carry no trackable energy
# deposition and produce no Cherenkov light — silently skip them.
_PDG_SKIP: frozenset = frozenset([
    1000080160,   # ¹⁶O nuclear remnant
    2000000002,   # GENIE nucleon-cluster pseudo-particle
    2000000101,   # GENIE "bindino" (nuclear binding-energy accounting)
])

# Map unsupported PDG codes to the closest fennel-supported equivalent.
# fennel knows: muons (±13), e/γ (±11, 22), π± (±211), K_long (130),
#               p/p̄ (±2212), n (2112).
_PDG_REMAP: dict = {
    # LeptonInjector pseudo-hadron (pre-existing patch, centralised here)
    -2000001006: 2212,
    # Charged kaons → K_long (same hadronic light-yield family)
    321: 130,    # K+
    -321: 130,   # K−
    310: 130,    # K_short
    311: 130,    # K⁰
    -311: 130,   # K̄⁰
    # Strange baryons → proton / neutron
    3112: 2212,  # Σ−
    3122: 2212,  # Λ⁰
    3212: 2112,  # Σ⁰
    3222: 2212,  # Σ+
    -3112: -2212, # Σ̄+
    -3122: -2212, # Λ̄⁰
    -3212: -2112, # Σ̄⁰
    -3222: -2212, # Σ̄−
    # Charmed mesons → pion / K_long
    411: 211,    # D+
    -411: -211,  # D−
    421: 130,    # D⁰
    -421: 130,   # D̄⁰
    # Charmed baryons → proton
    4122: 2212,  # Λc+
    -4122: -2212,
    4212: 2212,
    -4212: -2212,
    4222: 2212,  # Σc++
    -4222: -2212,
}


def _remap_pdg(particle_id: int) -> int:
    """Return the fennel-supported PDG code closest to *particle_id*."""
    return _PDG_REMAP.get(particle_id, particle_id)


def simple_cascade_light_yield(energy, *args):
    """Approximation for cascade light yield.

    Parameters
    ----------
    energy : float
        Particle energy in GeV.
    """
    photons_per_GeV = 5.3 * 250 * 1e2

    return energy * photons_per_GeV


def fennel_total_light_yield(energy, particle_id, wavelength_range):
    """Calculate total light yield using fennel.

    Parameters
    ----------
    energy : float
        Particle energy in GeV.
    particle_id : int
        Particle type (PDG ID).
    wavelength_range : tuple
        Wavelength interval (nm).
    """

    if particle_id in _PDG_SKIP:
        return 0.0
    particle_id = _remap_pdg(particle_id)
    funcs = fennel_instance.auto_yields(energy, particle_id, function=True)
    counts_func = funcs[0]

    wavelengths = jnp.linspace(wavelength_range[0], wavelength_range[1], 100)
    light_yield = jax_trapz(counts_func(energy, wavelengths).ravel(), wavelengths)

    return light_yield


def fennel_frac_long_light_yield(energy, particle_id, resolution=0.2):
    """Calculate the longitudinal light yield contribution.

    Integrate the longitudinal distribution in steps of ``resolution`` and
    return the relative contributions.

    Parameters
    ----------
    energy : float
        Particle energy in GeV.
    particle_id : int
        Particle type (PDG ID).
    resolution : float, optional
        Step length in m for evaluating the longitudinal distribution.
    """
    if particle_id in _PDG_SKIP:
        dummy_grid = jnp.arange(1e-3, 25, resolution)
        return jnp.zeros(len(dummy_grid) - 1), dummy_grid
    particle_id = _remap_pdg(particle_id)
    funcs = fennel_instance.auto_yields(energy, particle_id, function=True)
    long_func = funcs[4]
    int_grid = jnp.arange(1e-3, 25, resolution)

    def integrate(low, high, resolution=1000):
        trapz_x_eval = jnp.linspace(low, high, resolution) * 100  # to cm
        trapz_y_eval = long_func(energy, trapz_x_eval)
        return jax_trapz(trapz_y_eval, trapz_x_eval)

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
    """Create a pointlike light source.

    Parameters
    ----------
    pos : np.ndarray
        Cascade position (shape (3,)).
    t0 : float
        Cascade time.
    dir : np.ndarray
        Cascade direction (shape (3,)).
    energy : float
        Cascade energy.
    particle_id : int
        Particle type (PDG ID).
    wavelength_range : tuple, optional
        Wavelength interval (nm).

    Returns
    -------
    tuple
        Tuple ``(source_pos, source_dir, source_time, source_nphotons)`` where
        arrays are returned in JAX-friendly form for downstream propagation.
    """
    if particle_id in _PDG_SKIP:
        empty = jnp.empty((0, 3))
        return empty, empty, jnp.empty((0, 1)), jnp.empty((0, 1))
    particle_id = _remap_pdg(particle_id)
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
    """Create a realistic (elongated) particle cascade.

    The longitudinal profile is approximated by placing point-like light sources
    every ``resolution`` steps.

    Parameters
    ----------
    pos : np.ndarray
        Cascade position (shape (3,)).
    t0 : float
        Cascade time.
    dir : np.ndarray
        Cascade direction (shape (3,)).
    energy : float
        Cascade energy.
    particle_id : int
        Particle type (PDG ID).
    key : jax.random.PRNGKey
        Random key for stochastic operations.
    resolution : float, optional
        Step size for point-like light sources.
    moliere_rand : bool, optional
        If True, apply Molière randomization to lateral offsets.
    wavelength_range : tuple, optional
        Wavelength interval (nm).

    Returns
    -------
    tuple
        Tuple ``(source_pos, source_dir, source_time, source_nphotons)`` where
        arrays are returned in JAX-friendly form for downstream propagation.
    """
    if particle_id in _PDG_SKIP:
        empty = jnp.empty((0, 3))
        return empty, empty, jnp.empty((0, 1)), jnp.empty((0, 1))
    particle_id = _remap_pdg(particle_id)
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
        source_pos = dist_along[:, np.newaxis] * dir[np.newaxis, :] + pos[np.newaxis, :] + dpos_vec
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
