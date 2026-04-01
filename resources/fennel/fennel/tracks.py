# -*- coding: utf-8 -*-
"""
Track light yield calculations for charged particles.

This module implements the calculation of Cherenkov light yields from
charged particle tracks (primarily muons) using the Aachen parametrization.
Includes support for different energy loss mechanisms and angular distributions.
"""

import logging
import pickle
import pkgutil
from typing import Tuple, Union

import numpy as np

from .config import config

# Checking if JAX should be used
try:
    import jax as _jax
    import jax.numpy as jnp
    from jax import Array as JaxArray

    try:
        _jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
except ImportError:
    jnp = None
    JaxArray = None
    if config["general"]["jax"]:
        raise ImportError("JAX not found! Install with: pip install jax jaxlib")


_log = logging.getLogger(__name__)


class Track:
    """
    Track light yield calculator for charged particles.

    This class calculates Cherenkov light production from charged particle
    tracks, accounting for additional track length from energy loss mechanisms
    (ionization, bremsstrahlung, pair production, nuclear interactions) and
    angular distributions of emitted light.

    Attributes
    ----------
    _medium : str
        Name of the propagation medium ('water' or 'ice')
    _n : float
        Refractive index of the medium
    _params : dict
        Parametrization parameters loaded from data file
    additional_track_ratio : Callable
        Function to compute additional track length ratio
    cherenkov_angle_distro : Callable
        Function to compute Cherenkov angle distribution

    Methods
    -------
    additional_track_ratio(E, interaction)
        Calculate ratio of additional track length to original length
    cherenkov_angle_distro(energy, angles, n)
        Calculate angular distribution of Cherenkov light emission

    Examples
    --------
    >>> from fennel.tracks import Track
    >>> track = Track()
    >>> energies = np.array([10.0, 100.0, 1000.0])
    >>> ratio = track.additional_track_ratio(energies, 'total')
    >>> angles = np.linspace(0, 180, 100)
    >>> distro = track.cherenkov_angle_distro(100.0, angles, 1.333)

    Notes
    -----
    - Currently optimized for muon tracks
    - Supports both NumPy and JAX backends
    - JAX mode provides GPU acceleration for function mode
    - The Aachen parametrization is based on detailed GEANT4 simulations
    """

    def __init__(self) -> None:
        """
        Initialize the Track calculator.

        Loads parametrization data and sets up calculation methods
        based on whether JAX is enabled.

        Raises
        ------
        ValueError
            If the specified parametrization is not implemented
        ImportError
            If JAX mode is enabled but JAX is not installed
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.debug("Constructing a track object")
        self._medium = config["scenario"]["medium"]
        self._n = config["mediums"][self._medium]["refractive index"]
        if config["scenario"]["parametrization"] == "aachen":
            _log.info("Loading the aachen parametrization")
            param_file = pkgutil.get_data(
                __name__, "data/%s.pkl" % config["scenario"]["parametrization"]
            )
            self._params = pickle.loads(param_file)["track"]
        else:
            raise ValueError(
                "Track parametrization "
                + config["scenario"]["parametrization"]
                + " not implemented!"
            )
        if config["general"]["jax"]:
            _log.debug("Using JAX")
            self.additional_track_ratio = self._additional_track_ratio_fetcher_jax
            self.cherenkov_angle_distro = self._symmetric_angle_distro_fetcher_jax
        else:
            _log.debug("Using basic numpy")
            self.additional_track_ratio = self._additional_track_ratio_fetcher
            self.cherenkov_angle_distro = self._symmetric_angle_distro_fetcher

    ###########################################################################
    # Basic numpy
    def _additional_track_ratio_fetcher(
        self, E: Union[float, np.ndarray], interaction: str
    ) -> np.ndarray:
        """
        Calculate ratio of additional track length to original length.

        For a given energy and interaction type, computes the ratio between
        the additional track length produced by energy losses and the
        standard track length.

        Parameters
        ----------
        E : float or np.ndarray
            Particle energy in GeV. Can be scalar or array.
        interaction : {'ionization', 'brems', 'pair', 'nuclear', 'total'}
            Energy loss mechanism

        Returns
        -------
        ratio : np.ndarray
            Additional track length ratio. Values typically 0-1.

        Notes
        -----
        Uses parametrization: λ + κ * log(E)
        where λ (lambda) and κ (kappa) are medium and interaction dependent.
        """
        params = self._params["additional track " + self._medium][interaction]
        lambd = params["lambda"]
        kappa = params["kappa"]
        ratio = lambd + kappa * np.log(E)
        return ratio

    def _symmetric_angle_distro_fetcher(
        self, phi: np.ndarray, n: float, E: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate symmetric angular distribution of Cherenkov emission.

        Computes the angular distribution of emitted Cherenkov photons
        as a function of emission angle for given particle energy.

        Parameters
        ----------
        phi : np.ndarray
            Emission angles of interest in degrees.
            Shape: (n_angles,)
        n : float
            Refractive index of the medium. Must be > 1.
        E : float or np.ndarray
            Particle energy in GeV.

        Returns
        -------
        distro : np.ndarray
            Angular distribution of emitted photons.
            Shape matches phi. Normalized to integrate to 1.

        Notes
        -----
        - Assumes symmetric distribution around Cherenkov angle
        - Typical error < 10% compared to detailed simulations
        - TODO: Add asymmetry function for improved accuracy
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c}
        """
        a, b, c = self._energy_dependence_angle_pars(E)
        distro = np.array(
            [
                (a * np.exp(b * np.abs(1.0 / n - np.cos(np.deg2rad(phi_val))) ** c))
                for phi_val in phi
            ]
        )

        return distro

    def _energy_dependence_angle_pars(self, E):
        """Parametrizes the energy dependence of the angular distribution
        parameters

        Parameters
        ----------
        E : float / np.array
            The energies of interest

        Returns
        -------
        a : np.array
            The first parameter values for the given energies
        b : np.array
            The second parameter values for the given energies
        c : np.array
            The third parameter values for the given energies
        """
        params = self._params["angular distribution"]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        a = a_pars[0] * (np.log(E)) * a_pars[1]
        b = b_pars[0] * (np.log(E)) * b_pars[1]
        c = c_pars[0] * (np.log(E)) * c_pars[1]
        return a, b, c

    ###########################################################################
    # JAX
    def _additional_track_ratio_fetcher_jax(self, E: float, interaction: str) -> float:
        """Calculates the ratio between the additional track length
        and the original for a single energy. JAX implementation

        Parameters
        ----------
        E : float
            The energy of the particle in GeV
        interaction : str
            Name of the interaction

        Returns
        -------
        ratio : float
            The resulting ratio

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\lambda + \\kappa log(E)
        """
        params = self._params["additional track " + self._medium][interaction]
        lambd = params["lambda"]
        kappa = params["kappa"]
        ratio = lambd + kappa * jnp.log(E)
        return ratio

    def _symmetric_angle_distro_fetcher_jax(
        self, phi: float, n: float, E: float
    ) -> float:
        # TODO: Add asymmetry function
        """Calculates the symmetric angular distribution of the Cherenkov
        emission for a single energy. The error should lie below 10%.
        JAX implementation

        Parameters
        ----------
        phi : float
            The angles of interest in degrees
        n : float
            The refractive index
        E : float
            The energy of interest

        Returns
        -------
        distro : float
            The distribution of emitted photons given the angle. The
            result is a 2d array with the first axis for the angles and
            the second for the energies.

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c}
        """
        a, b, c = self._energy_dependence_angle_pars_jax(E)
        distro = a * jnp.exp(b * jnp.abs(1.0 / n - jnp.cos(jnp.deg2rad(phi))) ** c)
        return distro

    def _energy_dependence_angle_pars_jax(
        self, E: Union[float, "JaxArray"]
    ) -> Tuple["JaxArray", "JaxArray", "JaxArray"]:
        """
        Calculate energy-dependent angle parameters (JAX implementation).

        See Also
        --------
        _energy_dependence_angle_pars : NumPy version
        """
        params = self._params["angular distribution"]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        a = a_pars[0] * (jnp.log(E)) * a_pars[1]
        b = b_pars[0] * (jnp.log(E)) * b_pars[1]
        c = c_pars[0] * (jnp.log(E)) * c_pars[1]
        return a, b, c
