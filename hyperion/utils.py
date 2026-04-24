"""Utility functions."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from jax.lax import Precision
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma, gammaincc

from .constants import Constants


def calc_tres(t: np.ndarray, det_radius: float, det_dist: float, c_medium: float) -> np.ndarray:
    """Calculate time residual.

    The time residual is calculated by subtracting the expected (geometric)
    time a photon takes to travel ``det_dist``-``det_radius`` from the measured arrival time.

    Parameters
    ----------
    t : numpy.ndarray
        Measured arrival times.
    det_radius : float
        Detector radius.
    det_dist : float
        Distance to the detector.
    c_medium : float
        Speed of light in the medium.

    Returns
    -------
    tres : numpy.ndarray
        Time residual.
    """
    return t - ((det_dist - det_radius) / c_medium)


def cherenkov_ang_dist(costheta: np.ndarray, n_ph: float = 1.35) -> np.ndarray:
    """Angular distribution of Cherenkov photons for EM cascades.

    Taken from <https://arxiv.org/pdf/1210.5140.pdf>.

    Parameters
    ----------
    costheta : numpy.ndarray
        Cosine of the angle.
    n_ph : float, optional
        Refractive index for photons.

    Returns
    -------
    dist : numpy.ndarray
        Angular distribution values.
    """
    # params for e-

    a = Constants.CherenkovLightYield.AngDist.a
    b = Constants.CherenkovLightYield.AngDist.b
    c = Constants.CherenkovLightYield.AngDist.c
    cos_theta_c = 1 / n_ph
    d = Constants.CherenkovLightYield.AngDist.d
    return a * np.exp(b * np.abs(costheta - cos_theta_c) ** c) + d


def cherenkov_ang_dist_int(n_ph, lower=-1, upper=1):
    """Integral of the Cherenkov angular distribution function.

    Parameters
    ----------
    n_ph : float
        Refractive index for photons.
    lower : float, optional
        Lower bound of the integration interval.
    upper : float, optional
        Upper bound of the integration interval.

    Returns
    -------
    result : numpy.ndarray
        Definite integral of the angular distribution.
    """

    def incgamma(a, x):
        """Scaled upper incomplete gamma helper: Gamma(a) * gammaincc(a, x)."""
        return gamma(a) * gammaincc(a, x)

    a = Constants.CherenkovLightYield.AngDist.a
    b = Constants.CherenkovLightYield.AngDist.b
    c = Constants.CherenkovLightYield.AngDist.c
    d = Constants.CherenkovLightYield.AngDist.d
    n_ph = np.atleast_1d(n_ph)
    cos_theta_c = 1.0 / n_ph

    def indef_int(x):
        """Indefinite integral of the Cherenkov angular distribution.

        Parameters
        ----------
        x : float or array-like
            Integration variable.

        Returns
        -------
        numpy.ndarray
            Indefinite integral evaluated at ``x``.
        """

        def lower_branch(x, cos_theta_c):
            """Lower-branch expression for the indefinite integral.

            This branch applies for x < cos_theta_c.
            """
            return (
                1
                / c
                * (
                    c * d * x
                    + (a * (cos_theta_c - x) * incgamma(1 / c, -(b * (cos_theta_c - x) ** c)))
                    * (-(b * (cos_theta_c - x) ** c)) ** (-1 / c)
                )
            )

        def upper_branch(x, cos_theta_c):
            """Upper-branch expression for the indefinite integral."""
            return (
                1
                / c
                * (
                    c * d * x
                    + (a * (cos_theta_c - x) * incgamma(1 / c, -(b * (-cos_theta_c + x) ** c)))
                    * (-(b * (-cos_theta_c + x) ** c)) ** (-1 / c)
                )
            )

        peak_val = lower_branch(cos_theta_c - 1e-5, cos_theta_c)
        result = np.empty_like(cos_theta_c)
        mask = x < cos_theta_c
        result[mask] = lower_branch(x, cos_theta_c[mask])
        mask = x > cos_theta_c
        result[mask] = upper_branch(x, cos_theta_c[mask]) + 2 * peak_val[mask]

        return result

    return indef_int(upper) - indef_int(lower)


def calculate_min_number_steps(
    ref_index_func,
    scattering_length_function,
    det_dist,
    max_tres,
    wavelength,
    p_threshold,
):
    """Calculate the minimum number of steps required to satisfy ``p < p_threshold``.

    For a given refractive index function, scattering length function, distance to detector,
    and maximum time residual, calculate how many propagation steps have to be performed such
    that the probability of a photon having propagated for less time than the maximum time
    residual is less than ``p_threshold``.

    Parameters
    ----------
    ref_index_func : callable
        Function that returns the refractive index as a function of wavelength.
    scattering_length_function : callable
        Function that returns the scattering length as a function of wavelength.
    det_dist : float
        Distance from emitter to detector.
    max_tres : float
        Maximum time residual.
    wavelength : float
        Photon wavelength.
    p_threshold : float
        Probability threshold.

    Returns
    -------
    int
        Minimum number of steps required.
    """
    c_medium_f = lambda wl: (
        Constants.BaseConstants.c_vac
        / ref_index_func(  # noqa E731
            wl
        )
    )

    t_geo = det_dist / (c_medium_f(wavelength) / 1e9) + max_tres
    func = lambda step: (
        scipy.stats.gamma.cdf(
            t_geo * (c_medium_f(wavelength) / 1e9),
            step,
            scale=scattering_length_function(wavelength),
        )
        - 0.01
    )
    lim = scipy.optimize.brentq(func, 2, 100)

    return int(np.ceil(lim))


def make_cascadia_abs_len_func(sca_len_func):
    """Create an absorption-length function for Cascadia.

    Parameters
    ----------
    sca_len_func : callable
        Function mapping wavelength (nm) to scattering length.

    Returns
    -------
    callable
        Function mapping wavelength (nm) to absorption length.
    """
    att_lengths = np.asarray([[365, 10.4], [400, 14.6], [450, 27.7], [585, 7.1]])
    spl = UnivariateSpline(att_lengths[:, 0], np.log(att_lengths[:, 1]), k=2, s=0.01)

    def abs_len(wavelength):
        """Compute absorption length from interpolated attenuation and scattering.

        Parameters
        ----------
        wavelength : float or array-like
            Wavelength in nanometres.

        Returns
        -------
        float or numpy.ndarray
            Absorption length.
        """
        return 1 / (1 / np.exp(spl(wavelength)) - 1 / sca_len_func(wavelength))

    return abs_len


def rotate_to_new_direc(old_dir, new_dir, operand):
    """Rotate `operand` so that `old_dir` maps to `new_dir` using Rodrigues' formula.

    Parameters
    ----------
    old_dir : jax.numpy.ndarray
        Original direction vector (shape (3,) or (..., 3)).
    new_dir : jax.numpy.ndarray
        Target direction vector with same shape as `old_dir`.
    operand : jax.numpy.ndarray
        Vector(s) to rotate. The rotation is applied along the last axis.

    Returns
    -------
    jax.numpy.ndarray
        Rotated vector(s) with same shape as `operand`.
    """

    def _rotate(operand):
        """Rotate ``operand`` by the angle between ``old_dir`` and ``new_dir``.

        Parameters
        ----------
        operand : array-like
            Vector(s) to rotate.

        Returns
        -------
        array-like
            Rotated vector(s).
        """

        axis = jnp.cross(old_dir, new_dir)
        axis /= jnp.linalg.norm(axis)

        theta = jnp.arccos(jnp.dot(old_dir, new_dir, precision=Precision.HIGHEST))

        # Rodrigues' rotation formula

        v_rot = (
            operand * jnp.cos(theta)
            + jnp.cross(axis, operand) * jnp.sin(theta)
            + axis * jnp.dot(axis, operand, precision=Precision.HIGHEST) * (1 - jnp.cos(theta))
        )
        return v_rot

    v_rot = jax.lax.cond(jnp.all(old_dir == new_dir), lambda op: op, _rotate, operand)

    return v_rot


rotate_to_new_direc_v = jax.jit(jax.vmap(rotate_to_new_direc, in_axes=[None, None, 0]))
