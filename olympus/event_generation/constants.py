"""Physical Constants."""
import numpy as np

# Pandel: https://www.sciencedirect.com/science/article/pii/S0927650507001260


class Constants(object):
    """Collection of useful constants."""

    n_gr = 1.35634
    n_ph = 1.3195
    c_vac = 2.99792458e8 * 1e-9  # m/ns
    pandel_lambda = 33.3  # m
    pandel_rho = 0.004  # ns^-1
    lambda_abs = 98
    lambda_sca = 24
    theta_cherenkov = np.arccos(1 / n_ph)
    # 1 GeV EM cascade corresponds to 5.3 m Cherenkov track length
    # In the relevent wavelength interval, a single charged particle emmits 250 photons / cm
    photons_per_GeV = 5.3 * 250 * 1e2
