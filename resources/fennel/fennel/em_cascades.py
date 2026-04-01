# -*- coding: utf-8 -*-
"""
Electromagnetic cascade light yield calculations.

This module implements Cherenkov light yield calculations for electromagnetic
cascades initiated by electrons, positrons, or photons using the Aachen
parametrization. Includes longitudinal shower profiles and angular distributions.
"""

import logging
import pickle
import pkgutil
from typing import Tuple, Union

import numpy as np
from scipy.special import gamma as gamma_func

from .config import config

try:
    import jax as _jax
    import jax.numpy as jnp
    from jax import Array as JaxArray
    from jax.scipy.stats import gamma as jax_gamma

    try:
        _jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
except ImportError:
    jnp = None
    jax_gamma = None
    JaxArray = None
    if config["general"]["jax"]:
        raise ImportError("JAX not found! Install with: pip install jax jaxlib")

_log = logging.getLogger(__name__)


class EM_Cascade:
    """
    Electromagnetic cascade light yield calculator.

    Calculates Cherenkov light production from electromagnetic showers
    using the Aachen parametrization. Handles electron, positron, and
    photon-initiated cascades with longitudinal development profiles.

    Attributes
    ----------
    _medium : dict
        Medium properties dictionary
    _n : float
        Refractive index
    _radlength : float
        Radiation length in g/cm²
    _Lrad : float
        Radiation length in cm
    _params : dict
        Parametrization coefficients
    cherenkov_angle_distro : Callable
        Angular distribution calculator
    track_lengths : Callable
        Track length calculator
    long_profile : Callable
        Longitudinal profile calculator

    Examples
    --------
    >>> from fennel.em_cascades import EM_Cascade
    >>> em = EM_Cascade()
    >>> track_len, track_dev = em.track_lengths(100.0, particle=11)
    >>> z_grid = np.linspace(0, 1000, 100)
    >>> profile = em.long_profile(100.0, z_grid, particle=11)

    Notes
    -----
    - Parametrization based on GEANT4 simulations
    - Accounts for shower fluctuations
    - Supports both NumPy and JAX backends
    """

    def __init__(self) -> None:
        """
        Initialize the EM cascade calculator.

        Loads parametrization data and configures calculation methods
        based on JAX availability.

        Raises
        ------
        ValueError
            If parametrization is not implemented
        ImportError
            If JAX mode enabled but not installed
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.debug("Constructing an em cascade object")
        self._medium = config["mediums"][config["scenario"]["medium"]]
        self._n = self._medium["refractive index"]
        self._radlength = self._medium["radiation length"]
        self._Lrad = self._radlength / self._medium["density"]
        if config["scenario"]["parametrization"] == "aachen":
            _log.info("Loading the aachen parametrization")
            param_file = pkgutil.get_data(
                __name__, "data/%s.pkl" % config["scenario"]["parametrization"]
            )
            self._params = pickle.loads(param_file)["em cascade"]
        else:
            raise ValueError(
                "EM parametrization "
                + config["scenario"]["parametrization"]
                + " not implemented!"
            )
        if config["general"]["jax"]:
            _log.info("Running with JAX functions")
            self.cherenkov_angle_distro = self._symmetric_angle_distro_jax
            self.track_lengths = self._track_lengths_fetcher_jax
            self.long_profile = self._log_profile_func_fetcher_jax
        else:
            _log.info("Running with basic functions")
            self.cherenkov_angle_distro = self._symmetric_angle_distro
            self.track_lengths = self._track_lengths_fetcher
            self.long_profile = self._log_profile_func_fetcher

    ###########################################################################
    # Numpy
    def _track_lengths_fetcher(
        self, E: Union[float, np.ndarray], name: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate EM cascade track lengths and fluctuations.

        Parametrizes the energy-dependent track length distribution for
        electromagnetic cascades using fitted parameters from simulations.

        Parameters
        ----------
        E : float or np.ndarray
            Cascade energy in GeV
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        track_length : np.ndarray
            Mean track length in cm
        track_length_dev : np.ndarray
            Standard deviation of track length in cm

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\alpha E^{\\beta}
        """
        params = self._params["track parameters"][name]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def _log_profile_func_fetcher(
        self, E: Union[float, np.ndarray], z: np.ndarray, name: int
    ) -> np.ndarray:
        """
        Calculate longitudinal shower profile distribution.

        Parametrizes the cascade development along the shower axis
        using a gamma distribution with energy-dependent parameters.

        Parameters
        ----------
        E : float or np.ndarray
            Cascade energy in GeV
        z : np.ndarray
            Cascade depth in cm
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        res : np.ndarray
            Normalized longitudinal profile l^(-1) * dl/dt [1/cm]
            Shape: (n_energies, n_depths)

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b\\times \\frac{(tb)^{a-1}e^{-tb}}{\\Gamma(a)}

        where t = z/L_rad, with L_rad the radiation length.
        """
        t = z / self._Lrad
        b = self._b_energy_fetch(name)
        a = self._a_energy_fetch(E, name)
        a = np.array(a).flatten()
        res = np.array(
            [
                b * ((t * b) ** (a_val - 1.0) * np.exp(-(t * b)) / gamma_func(a_val))
                for a_val in a
            ]
        )
        return res

    def _a_energy_fetch(self, E: Union[float, np.ndarray], name: int) -> np.ndarray:
        """
        Calculate energy-dependent 'a' parameter for longitudinal profile.

        The 'a' parameter controls the shape of the gamma distribution
        used for the longitudinal shower profile.

        Parameters
        ----------
        E : float or np.ndarray
            Cascade energy in GeV
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        a : np.ndarray
            Shape parameter for gamma distribution (dimensionless)

        Notes
        -----
        Fitted from GEANT4 simulations of electromagnetic showers.
        The analytical form of the parametrization is:

        .. math:: \\alpha + \\beta log_{10}(E)
        """
        params = self._params["longitudinal parameters"][name]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * np.log10(E)
        return a

    def _b_energy_fetch(self, name: int) -> float:
        """
        Get constant 'b' parameter for longitudinal profile.

        The 'b' parameter is the scale parameter for the gamma distribution.
        Currently assumed energy-independent.

        Parameters
        ----------
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        b : float
            Scale parameter for gamma distribution (dimensionless)

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b = b
        """
        params = self._params["longitudinal parameters"][name]
        b = params["b"]
        return b

    def _symmetric_angle_distro(
        self, phi: np.ndarray, n: float, name: int
    ) -> np.ndarray:
        """
        Calculate symmetric angular distribution of Cherenkov photons.

        Parametrizes the angular emission pattern of Cherenkov light
        from electromagnetic cascades.

        Parameters
        ----------
        phi : np.ndarray
            Emission angles in degrees
        n : float
            Refractive index of medium
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        distro : np.ndarray
            Angular distribution [photons/degree]

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c} + d

        TODO: Add asymmetry function
        TODO: Add changes with shower depth
        Error typically below 10%.
        """
        params = self._params["angular distribution"][name]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        d = params["d"]
        distro = a * np.exp(b * np.abs(1.0 / n - np.cos(np.deg2rad(phi))) ** c) + d
        return distro

    ###########################################################################
    # JAX
    def _track_lengths_fetcher_jax(
        self, E: Union[float, "JaxArray"], name: int
    ) -> Tuple["JaxArray", "JaxArray"]:
        """
        Calculate EM cascade track lengths (JAX implementation).

        JAX-accelerated version of track length calculation for GPU execution.

        Parameters
        ----------
        E : float or jax.Array
            Cascade energy in GeV
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        track_length : jax.Array
            Mean track length in cm
        track_length_dev : jax.Array
            Standard deviation of track length in cm

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\alpha E^{\\beta}

        See Also
        --------
        _track_lengths_fetcher : NumPy version
        """
        params = self._params["track parameters"][name]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def _log_profile_func_fetcher_jax(
        self, E: Union[float, "JaxArray"], z: Union[float, "JaxArray"], name: int
    ) -> "JaxArray":
        """
        Calculate longitudinal shower profile (JAX implementation).

        JAX-accelerated version for GPU execution.

        Parameters
        ----------
        E : float or jax.Array
            Cascade energy in GeV
        z : float or jax.Array
            Cascade depth in cm
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        res : jax.Array
            Normalized longitudinal profile l^(-1) * dl/dt [1/cm]

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b\\times \\frac{(tb)^{a-1}e^{-tb}}{\\Gamma(a)}

        See Also
        --------
        _log_profile_func_fetcher : NumPy version
        """
        t = z / self._Lrad
        b = self._b_energy_fetch_jax(name)
        a = self._a_energy_fetch_jax(E, name)
        res = jax_gamma.pdf(t * b, a) * b
        return res

    def _a_energy_fetch_jax(self, E: Union[float, "JaxArray"], name: int) -> "JaxArray":
        """
        Calculate energy-dependent 'a' parameter (JAX implementation).

        JAX-accelerated version of shape parameter calculation.

        Parameters
        ----------
        E : float or jax.Array
            Cascade energy in GeV
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        a : jax.Array
            Shape parameter for gamma distribution

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\alpha + \\beta log_{10}(E)

        See Also
        --------
        _a_energy_fetch : NumPy version
        """
        params = self._params["longitudinal parameters"][name]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * jnp.log10(E)
        return a

    def _b_energy_fetch_jax(self, name: int) -> float:
        """
        Get constant 'b' parameter (JAX implementation).

        JAX-accelerated version for scale parameter.

        Parameters
        ----------
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        b : float
            Scale parameter for gamma distribution

        See Also
        --------
        _b_energy_fetch : NumPy version

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b = b
        """
        params = self._params["longitudinal parameters"][name]
        b = params["b"]
        return b

    def _symmetric_angle_distro_jax(
        self, phi: Union[float, "JaxArray"], n: float, name: int
    ) -> "JaxArray":
        """
        Calculate symmetric angular distribution (JAX implementation).

        JAX-accelerated version of angular distribution calculation.

        Parameters
        ----------
        phi : float
            The angles of interest in degrees
        n : float
            The refractive index
        name : int
            The particle of interest

        Returns
        -------
        JAX-accelerated version of angular distribution calculation.

        Parameters
        ----------
        phi : float or jax.Array
            Emission angles in degrees
        n : float
            Refractive index of medium
        name : int
            PDG particle code (11=e±, 22=γ)

        Returns
        -------
        distro : jax.Array
            Angular distribution [photons/degree]

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c} + d

        TODO: Add asymmetry function
        TODO: Add changes with shower depth

        See Also
        --------
        _symmetric_angle_distro : NumPy version
        """
        params = self._params["angular distribution"][name]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        d = params["d"]
        distro = a * jnp.exp(b * jnp.abs(1.0 / n - jnp.cos(jnp.deg2rad(phi))) ** c) + d
        return distro
