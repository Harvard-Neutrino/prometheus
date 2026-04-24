"""
Physical constants for event generation.

Notes
-----
Pandel: https://www.sciencedirect.com/science/article/pii/S0927650507001260
"""
import numpy as np

class Constants(object):
    """
    Collection of useful constants.

    Attributes
    ----------
    n_gr : float
        Group refractive index.
    n_ph : float
        Phase refractive index.
    c_vac : float
        Speed of light in medium units (m/ns).
    pandel_lambda : float
        Pandel lambda parameter (m).
    pandel_rho : float
        Pandel rho parameter (ns^-1).
    lambda_abs : float
        Absorption length (m).
    lambda_sca : float
        Scattering length (m).
    theta_cherenkov : float
        Cherenkov angle in radians.
    photons_per_GeV : float
        Number of photons per GeV of EM cascade energy.

    Notes
    -----
    1 GeV EM cascade corresponds to 5.3 m Cherenkov track length. In the
    relevant wavelength interval, a single charged particle emits 250 photons/cm.
    """

    n_gr = 1.35634
    n_ph = 1.3195
    c_vac = 2.99792458e8 * 1e-9  # m/ns
    pandel_lambda = 33.3  # m
    pandel_rho = 0.004  # ns^-1
    lambda_abs = 98
    lambda_sca = 24
    theta_cherenkov = np.arccos(1 / n_ph)
    photons_per_GeV = 5.3 * 250 * 1e2
