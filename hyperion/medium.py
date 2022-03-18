"""This module hosts a collection of functions related to the optical properties of a medium."""
import jax.numpy as jnp
from jax import random
import jax
from jax.lax import cond


def henyey_greenstein_scattering_angle(key, g=0.9):
    """Henyey-Greenstein scattering in one plane."""
    eta = random.uniform(key)
    costheta = (
        1 / (2 * g) * (1 + g ** 2 - ((1 - g ** 2) / (1 + g * (2 * eta - 1))) ** 2)
    )
    return jnp.arccos(costheta)


def rayleigh_scattering_angle(key):
    """Rayleigh scattering. Adapted from clsim."""
    b = 0.835
    p = 1.0 / 0.835

    q = (b + 3.0) * ((random.uniform(key)) - 0.5) / b
    d = q * q + p * p * p

    u1 = -q + jnp.sqrt(d)
    u = jnp.cbrt(jnp.abs(u1)) * jnp.sign(u1)

    v1 = -q - jnp.sqrt(d)
    v = jnp.cbrt(jnp.abs(v1)) * jnp.sign(v1)

    return jnp.arccos(jax.lax.clamp(-1.0, u + v, 1.0))


def liu_scattering_angle(key, g=0.95):
    """
    Simplified liu scattering.

    https://arxiv.org/pdf/1301.5361.pdf
    """
    beta = (1 - g) / (1 + g)
    xi = random.uniform(key)
    costheta = 2 * xi ** beta - 1
    return jnp.arccos(costheta)


def make_mixed_scattering_func(f1, f2, ratio):
    """
    Create a mixture model with two sampling functions.

    Paramaters:
        f1, f2: functions
            Sampling functions taking one argument (random key)
        ratio: float
            Fraction of samples drawn from f1
    """

    def _f(key):
        k1, k2 = random.split(key)
        is_f1 = random.uniform(k1) < ratio

        return cond(is_f1, f1, f2, k2)

    return _f


"""Mix of HG and Rayleigh. Distribution similar to ANTARES Petzold+Rayleigh."""
mixed_hg_rayleigh_antares = make_mixed_scattering_func(
    rayleigh_scattering_angle,
    lambda k: henyey_greenstein_scattering_angle(k, 0.97),
    0.15,
)

"""Mix of HG and Liu. IceCube"""
mixed_hg_liu_icecube = make_mixed_scattering_func(
    lambda k: liu_scattering_angle(k, 0.95),
    lambda k: henyey_greenstein_scattering_angle(k, 0.95),
    0.35,
)


def make_wl_dep_sca_len_func(vol_conc_small_part, vol_conc_large_part):
    """
    Make a function that calculates the scattering length based on particle concentrations.
    Copied from clsim.

    Parameters:
        vol_conc_small_part: Volumetric concentration of small particles (ppm)
        vol_conc_small_part: Volumetric concentration of large particles (ppm)

    """

    def sca_len(wavelength):
        ref_wlen = 550  # nm
        x = ref_wlen / wavelength

        sca_coeff = (
            0.0017 * jnp.power(x, 4.3)
            + 1.34 * vol_conc_small_part * jnp.power(x, 1.7)
            + 0.312 * vol_conc_large_part * jnp.power(x, 0.3)
        )

        return 1 / sca_coeff

    return sca_len


sca_len_func_antares = make_wl_dep_sca_len_func(0.0075, 0.0075)


def make_ref_index_func(salinity, temperature, pressure):
    """
    Make function that returns refractive index as function of wavelength.

    Parameters:
        salinity: float
            Salinity in parts per thousand
        temperature: float
            Temperature in C
        pressure: float
            Pressure in bar
    """
    n0 = 1.31405
    n1 = 1.45e-5
    n2 = 1.779e-4
    n3 = 1.05e-6
    n4 = 1.6e-8
    n5 = 2.02e-6
    n6 = 15.868
    n7 = 0.01155
    n8 = 0.00423
    n9 = 4382
    n10 = 1.1455e6

    a01 = (
        n0
        + (n2 - n3 * temperature + n4 * temperature * temperature) * salinity
        - n5 * temperature * temperature
        + n1 * pressure
    )
    a2 = n6 + n7 * salinity - n8 * temperature
    a3 = -n9
    a4 = n10

    def ref_index_func(wavelength):

        x = 1 / wavelength
        return a01 + x * (a2 + x * (a3 + x * a4))

    return ref_index_func


antares_ref_index_func = make_ref_index_func(
    pressure=215.82225 / 1.01325, temperature=13.1, salinity=38.44
)

cascadia_ref_index_func = make_ref_index_func(
    pressure=269.44088 / 1.01325, temperature=1.8, salinity=34.82
)

medium_collections = {
    "pone": (cascadia_ref_index_func, mixed_hg_rayleigh_antares, sca_len_func_antares)
}
