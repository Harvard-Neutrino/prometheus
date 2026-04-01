# -*- coding: utf-8 -*-
"""
Hadronic cascade light yield calculations.

This module implements Cherenkov light yield calculations for hadronic
showers initiated by charged pions, kaons, protons, or neutrons using
the Aachen parametrization. Includes electromagnetic fraction modeling,
muon production, and longitudinal development.
"""

import logging
import pickle
import pkgutil
from typing import Tuple, Union

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma as gamma_func

from .config import config
from .particle import Particle

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


class Hadron_Cascade:
    """
    Hadronic cascade light yield calculator.

    Calculates Cherenkov light production from hadronic showers using
    the Aachen parametrization. Handles charged pions, kaons, protons,
    and neutrons with electromagnetic fraction and muon production.

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
    em_fraction : Callable
        EM fraction calculator
    muon_production : Callable
        Muon production calculator

    Examples
    --------
    >>> from fennel.hadron_cascades import Hadron_Cascade
    >>> had = Hadron_Cascade()
    >>> track_len, track_dev = had.track_lengths(100.0, particle=211)
    >>> em_frac = had.em_fraction(100.0, particle=211)

    Notes
    -----
    - Parametrization based on GEANT4 simulations
    - Accounts for EM fraction and muon production
    - Supports both NumPy and JAX backends
    """

    def __init__(self) -> None:
        """
        Initialize the hadronic cascade calculator.

        Loads parametrization data and muon production splines,
        configures calculation methods based on JAX availability.

        Raises
        ------
        ValueError
            If parametrization is not implemented or muon data not found
        ImportError
            If JAX mode enabled but not installed
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.debug("Constructing a hadron cascade object")
        self._medium = config["mediums"][config["scenario"]["medium"]]
        self._n = self._medium["refractive index"]
        self._radlength = self._medium["radiation length"]
        self._Lrad = self._radlength / self._medium["density"]
        if config["scenario"]["parametrization"] == "aachen":
            _log.info("Loading the aachen parametrization")
            param_file = pkgutil.get_data(
                __name__, "data/%s.pkl" % config["scenario"]["parametrization"]
            )
            self._params = pickle.loads(param_file)["hadron cascade"]
            muon_data = pkgutil.get_data(
                __name__,
                "data/%s_muon_production.pkl" % (config["scenario"]["parametrization"]),
            )
            if muon_data is None:
                raise ValueError("Muon production data not found!")
            self.__muon_prod_dict = pickle.loads(muon_data)
            _log.debug("Constructing spline dictionary")
            self.__muon_prod_spl_pars = {}
            for pdg_id in config["simulation"]["hadron particles"]:
                self.__muon_prod_spl_pars[pdg_id] = {
                    "alpha": UnivariateSpline(
                        self.__muon_prod_dict[pdg_id][0],
                        self.__muon_prod_dict[pdg_id][1],
                        k=1,
                        s=0,
                        ext=3,
                    ),
                    "beta": UnivariateSpline(
                        self.__muon_prod_dict[pdg_id][0],
                        self.__muon_prod_dict[pdg_id][2],
                        k=1,
                        s=0,
                        ext=3,
                    ),
                    "gamma": UnivariateSpline(
                        self.__muon_prod_dict[pdg_id][0],
                        self.__muon_prod_dict[pdg_id][3],
                        k=1,
                        s=0,
                        ext=3,
                    ),
                }
        else:
            raise ValueError(
                "Hadronic parametrization "
                + config["scenario"]["parametrization"]
                + " not implemented!"
            )
        if config["general"]["jax"]:
            _log.info("Running with JAX functions")
            self.cherenkov_angle_distro = self._symmetric_angle_distro_jax
            self.track_lengths = self._track_lengths_fetcher_jax
            self.em_fraction = self._em_fraction_fetcher_jax
            self.long_profile = self._log_profile_func_fetcher_jax
            self.muon_production = self._muon_production_fetcher_jax
        else:
            _log.info("Running with basic functions")
            self.cherenkov_angle_distro = self._symmetric_angle_distro
            self.track_lengths = self._track_lengths_fetcher
            self.em_fraction = self._em_fraction_fetcher
            self.long_profile = self._log_profile_func_fetcher
            self.muon_production = self._muon_production_fetcher

    ###########################################################################
    # Numpy
    def _track_lengths_fetcher(
        self, E: Union[float, np.ndarray], particle: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate hadronic cascade track lengths and fluctuations.

        Parametrizes the energy-dependent track length distribution for
        hadronic cascades using fitted parameters from simulations.

        Parameters
        ----------
        E : float or np.ndarray
            Cascade energy in GeV
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)

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
        params = self._params["track parameters"][particle]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def _em_fraction_fetcher(
        self, E: Union[float, np.ndarray], particle: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate electromagnetic fraction of hadronic shower.

        Returns the fraction of shower energy going into electromagnetic
        cascades and its fluctuation.

        Parameters
        ----------
        E : float or np.ndarray
            Hadronic shower energy in GeV
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)

        Returns
        -------
        em_fraction : np.ndarray
            Mean electromagnetic fraction (0 to 1)
        em_fraction_sd : np.ndarray
            Standard deviation of EM fraction

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: 1 - (1 - f_0)\\left(\\frac{E}{E_s}\\right)^{-m}
        """
        params = self._params["em fraction"][particle]
        Es = params["Es"]
        f0 = params["f0"]
        m = params["m"]
        sigma0 = params["sigma0"]
        gamma = params["gamma"]
        em_fraction = 1.0 - (1.0 - f0) * (E / Es) ** (-m)
        em_fraction_sd = sigma0 * np.log(E) ** (-gamma)
        return em_fraction, em_fraction_sd

    def _log_profile_func_fetcher(
        self, E: Union[float, np.ndarray], z: np.ndarray, particle: int
    ) -> np.ndarray:
        """
        Calculate longitudinal hadronic shower profile distribution.

        Parametrizes the cascade development along the shower axis
        using a gamma distribution with energy-dependent parameters.

        Parameters
        ----------
        E : float or np.ndarray
            Cascade energy in GeV
        z : np.ndarray
            Cascade depth in cm
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)

        Returns
        -------
        res : np.ndarray
            Normalized longitudinal profile l^(-1) * dl/dt [1/cm]
            Shape: (n_energies, n_depths)

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b\\times \\frac{(tb)^{a-1}e^{-tb}}{\\Gamma(a)}
        """
        t = z / self._Lrad
        b = self._b_energy_fetcher(particle)
        a = self._a_energy_fetcher(E, particle)
        a = np.array([a]).flatten()
        # gamma.pdf seems far slower than the explicit implementation
        res = np.array(
            [
                b * ((t * b) ** (a_val - 1.0) * np.exp(-(t * b)) / gamma_func(a_val))
                for a_val in a
            ]
        )
        return res

    def _a_energy_fetcher(
        self, E: Union[float, np.ndarray], particle: int
    ) -> np.ndarray:
        """
        Calculate energy-dependent 'a' parameter for longitudinal profile.

        The 'a' parameter controls the shape of the gamma distribution
        used for the longitudinal shower profile.

        Parameters
        ----------
        E : float or np.ndarray
            Cascade energy in GeV
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)

        Returns
        -------
        a : np.ndarray
            Shape parameter for gamma distribution (dimensionless)

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\alpha + \\beta log_{10}(E)
        """
        params = self._params["longitudinal parameters"][particle]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * np.log10(E)
        return a

    def _b_energy_fetcher(self, particle: int) -> float:
        """
        Get constant 'b' parameter for longitudinal profile.

        The 'b' parameter is the scale parameter for the gamma distribution.
        Currently assumed energy-independent.

        Parameters
        ----------
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)

        Returns
        -------
        b : float
            Scale parameter for gamma distribution (dimensionless)

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b = b
        """
        params = self._params["longitudinal parameters"][particle]
        b = params["b"]
        return b

    def _symmetric_angle_distro(
        self, E: Union[float, np.ndarray], phi: np.ndarray, n: float, particle: int
    ) -> np.ndarray:
        """
        Calculate symmetric angular distribution of Cherenkov photons.

        Parametrizes the angular emission pattern of Cherenkov light
        from hadronic cascades with energy-dependent parameters.

        Parameters
        ----------
        E : float or np.ndarray
            Cascade energy in GeV
        phi : np.ndarray
            Emission angles in degrees
        n : float
            Refractive index of medium
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)

        Returns
        -------
        distro : np.ndarray
            Angular distribution [photons/degree]
            Shape: (n_angles, n_energies)

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c} + d

        TODO: Add asymmetry function
        Error typically below 10%.
        """
        a, b, c, d = self._energy_dependence_angle_pars(E, particle)
        distro = np.array(
            [
                (a * np.exp(b * np.abs(1.0 / n - np.cos(np.deg2rad(phi_val))) ** c) + d)
                for phi_val in phi
            ]
        )

        return np.nan_to_num(distro)

    def _energy_dependence_angle_pars(
        self, E: Union[float, np.ndarray], particle: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate energy-dependent angular distribution parameters.

        Parametrizes how the angular distribution parameters (a, b, c, d)
        vary with cascade energy.

        Parameters
        ----------
        E : float or np.ndarray
            Cascade energy(ies) in GeV
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)

        Returns
        -------
        a : np.ndarray
            First parameter values for given energies
        b : np.ndarray
            Second parameter values for given energies
        c : np.ndarray
            Third parameter values for given energies
        d : np.ndarray
            Fourth parameter values for given energies

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: par = par_0 \\, log(E)^{par_1}
        """
        params = config["hadron cascade"]["angular distribution"][particle]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        d_pars = params["d pars"]
        a = a_pars[0] * (np.log(E)) ** a_pars[1]
        b = b_pars[0] * (np.log(E)) ** b_pars[1]
        c = c_pars[0] * (np.log(E)) ** c_pars[1]
        d = d_pars[0] * (np.log(E)) ** d_pars[1]
        return (
            np.array([a]).flatten(),
            np.array([b]).flatten(),
            np.array([c]).flatten(),
            np.array([d]).flatten(),
        )

    def _muon_production_fetcher(
        self,
        Eprim: Union[float, np.ndarray],
        Emu: Union[float, np.ndarray],
        particle: int,
    ) -> np.ndarray:
        """
        Calculate muon production distribution in hadronic cascades.

        Parametrizes the energy distribution of muons produced in
        hadronic showers.

        Parameters
        ----------
        Eprim : float or np.ndarray
            Primary particle energy(ies) in GeV
        Emu : float or np.ndarray
            Muon energy(ies) in GeV
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)

        Returns
        -------
        distro : np.ndarray
            Muon production distribution [dN/dE]

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: -\\alpha + \\beta\\left(\\frac{E}{GeV}\\right)^{-\\gamma}
        """
        # Converting the primary energy to an array
        energy_prim = np.array([Eprim]).flatten()
        # Converting to np.array for ease of use
        energy = np.array([Emu]).flatten()
        alpha, beta, gamma = self._muon_production_pars(energy_prim, particle)
        # Removing too small values
        energy[energy <= 1.0] = 0.0
        # Removing all secondary energies above the primary energy(ies)
        energy_2d = np.array([energy for _ in range(len(energy_prim))])
        # Removing energies above the primary
        distro = []
        for id_arr, _ in enumerate(energy_2d):
            energy_2d[id_arr][energy_2d[id_arr] > energy_prim[id_arr]] = 0.0
            # Removing too large values
            energy[
                energy > (alpha[id_arr] / beta[id_arr]) ** (-1.0 / gamma[id_arr])
            ] = 0.0
            distro.append(
                energy_prim[id_arr]
                * (
                    -alpha[id_arr]
                    + beta[id_arr] * (energy_2d[id_arr] ** (-gamma[id_arr]))
                )
            )
        distro = np.array(distro)
        distro[distro == np.inf] = 0.0
        distro[distro < 0.0] = 0.0
        return distro

    def _muon_production_pars(self, E, particle: Particle):
        """Constructs the parametrization values for the energies of interest.

        Parameters
        ----------
        E : float/np.array
            The energy(ies) of interest
        particle: Particle
            The particle of interest

        Returns
        -------
        alpha : float/np.array
            The first parameter values for the given energies
        beta : float/np.array
            The second parameter values for the given energies
        gamma : float/np.array
            The third parameter values for the given energies
        """
        alpha = self.__muon_prod_spl_pars[particle._pdg_id]["alpha"](E)
        beta = self.__muon_prod_spl_pars[particle._pdg_id]["beta"](E)
        gamma = self.__muon_prod_spl_pars[particle._pdg_id]["gamma"](E)
        return alpha, beta, gamma

    ###########################################################################
    # JAX
    def _track_lengths_fetcher_jax(
        self, E: Union[float, "JaxArray"], particle: int
    ) -> Tuple["JaxArray", "JaxArray"]:
        """
        Calculate hadronic cascade track lengths (JAX implementation).

        JAX-accelerated version for GPU execution.

        Parameters
        ----------
        E : float or jax.Array
            Cascade energy in GeV
        particle : int
            PDG particle code

        Returns
        -------
        track_length : jax.Array
            Mean track length in cm
        track_length_dev : jax.Array
            Standard deviation in cm

        See Also
        --------
        _track_lengths_fetcher : NumPy version
        """
        params = self._params["track parameters"][particle]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def _em_fraction_fetcher_jax(
        self, E: Union[float, "JaxArray"], particle: int
    ) -> Tuple["JaxArray", "JaxArray"]:
        """
        Calculate electromagnetic fraction (JAX implementation).

        JAX-accelerated version for GPU execution.

        Parameters
        ----------
        E : float or jax.Array
            Hadronic shower energy in GeV
        particle : int
            PDG particle code

        Returns
        -------
        em_fraction : jax.Array
            Mean electromagnetic fraction
        em_fraction_sd : jax.Array
            Standard deviation

        See Also
        --------
        _em_fraction_fetcher : NumPy version
        """
        params = self._params["em fraction"][particle]
        Es = params["Es"]
        f0 = params["f0"]
        m = params["m"]
        sigma0 = params["sigma0"]
        gamma = params["gamma"]
        em_fraction = 1.0 - (1.0 - f0) * (E / Es) ** (-m)
        em_fraction_sd = sigma0 * jnp.log(E) ** (-gamma)
        return em_fraction, em_fraction_sd

    def _log_profile_func_fetcher_jax(
        self, E: Union[float, "JaxArray"], z: Union[float, "JaxArray"], particle: int
    ) -> "JaxArray":
        """
        Calculate longitudinal profile (JAX implementation).

        JAX-accelerated version for GPU execution.

        Parameters
        ----------
        E : float or jax.Array
            Cascade energy in GeV
        z : float or jax.Array
            Cascade depth in cm
        particle : int
            PDG particle code

        Returns
        -------
        res : jax.Array
            Normalized longitudinal profile [1/cm]

        See Also
        --------
        _log_profile_func_fetcher : NumPy version
        """
        t = z / self._Lrad
        b = self._b_energy_fetcher_jax(particle)
        a = self._a_energy_fetcher_jax(E, particle)
        res = jax_gamma.pdf(t * b, a) * b
        return res

    def _a_energy_fetcher_jax(
        self, E: Union[float, "JaxArray"], particle: int
    ) -> "JaxArray":
        """
        Calculate 'a' parameter (JAX implementation).

        See Also
        --------
        _a_energy_fetcher : NumPy version
        """
        params = self._params["longitudinal parameters"][particle]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * jnp.log10(E)
        return a

    def _b_energy_fetcher_jax(self, particle: int) -> float:
        """
        Get 'b' parameter (JAX implementation).

        See Also
        --------
        _b_energy_fetcher : NumPy version
        """
        params = self._params["longitudinal parameters"][particle]
        b = params["b"]
        return b

    def _symmetric_angle_distro_jax(
        self,
        E: Union[float, "JaxArray"],
        phi: Union[float, "JaxArray"],
        n: float,
        particle: int,
    ) -> "JaxArray":
        """
        Calculate symmetric angular distribution (JAX implementation).

        JAX-accelerated version for GPU execution.

        Parameters
        ----------
        E : float or jax.Array
            Cascade energy in GeV
        phi : float or jax.Array
            Emission angles in degrees
        n : float
            Refractive index
        particle : int
            PDG particle code

        Returns
        -------
        distro : jax.Array
            Angular distribution [photons/degree]

        See Also
        --------
        _symmetric_angle_distro : NumPy version
        """
        a, b, c, d = self._energy_dependence_angle_pars_jax(E, particle)
        distro = a * jnp.exp(b * jnp.abs(1.0 / n - jnp.cos(jnp.deg2rad(phi))) ** c) + d

        return jnp.nan_to_num(distro)

    def _energy_dependence_angle_pars_jax(
        self, E: Union[float, "JaxArray"], particle: int
    ) -> Tuple["JaxArray", "JaxArray", "JaxArray", "JaxArray"]:
        """
        Calculate energy-dependent angle parameters (JAX implementation).

        See Also
        --------
        _energy_dependence_angle_pars : NumPy version
        """
        params = self._params["angular distribution"][particle]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        d_pars = params["d pars"]
        a = a_pars[0] * (jnp.log(E)) ** a_pars[1]
        b = b_pars[0] * (jnp.log(E)) ** b_pars[1]
        c = c_pars[0] * (jnp.log(E)) ** c_pars[1]
        d = d_pars[0] * (jnp.log(E)) ** d_pars[1]
        return (a, b, c, d)

    def _muon_production_fetcher_jax(
        self,
        Eprim: Union[float, "JaxArray"],
        Emu: Union[float, "JaxArray"],
        particle: int,
    ) -> "JaxArray":
        """
        Calculate muon production distribution (JAX implementation).

        JAX-accelerated version for GPU execution.

        Parameters
        ----------
        Eprim : float or jax.Array
            Primary particle energy in GeV
        Emu : float or jax.Array
            Muon energy in GeV
        particle : int
            PDG particle code

        Returns
        -------
        distro : jax.Array
            Muon production distribution [dN/dE]

        See Also
        --------
        _muon_production_fetcher : NumPy version
        """
        energy_prim = Eprim
        energy = Emu
        alpha, beta, gamma = self._muon_production_pars_jax(energy_prim, particle)
        # Removing too small values
        if Emu < 1.0:
            return 0.0
        # Removing all secondary energies above the primary energy(ies)
        if Emu >= energy_prim:
            return 0.0
        # Removing too large values
        if Emu > (alpha / beta) ** (-1.0 / gamma):
            return 0.0
        distro = energy_prim * (-alpha + beta * (energy ** (-gamma)))
        # Removing numerical errors
        if distro == np.inf:
            return 0.0
        if distro < 0.0:
            return 0.0
        return distro

    def _muon_production_pars_jax(
        self, E: Union[float, "JaxArray"], particle: int
    ) -> Tuple["JaxArray", "JaxArray", "JaxArray"]:
        """
        Get muon production parameters (JAX implementation).

        See Also
        --------
        _muon_production_pars : NumPy version
        """
        alpha = self.__muon_prod_spl_pars[particle]["alpha"](E)
        beta = self.__muon_prod_spl_pars[particle]["beta"](E)
        gamma = self.__muon_prod_spl_pars[particle]["gamma"](E)
        return alpha, beta, gamma
