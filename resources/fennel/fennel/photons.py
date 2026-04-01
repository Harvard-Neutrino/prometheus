# -*- coding: utf-8 -*-
"""
Cherenkov photon yield calculations.

This module calculates the number and distribution of Cherenkov photons
produced by particles (tracks) and cascades (electromagnetic and hadronic).
Provides fetcher and builder methods for different particle types.
"""

import logging
from typing import Callable, Dict, Tuple, Union

import numpy as np

from .config import config
from .em_cascades import EM_Cascade
from .hadron_cascades import Hadron_Cascade
from .particle import Particle
from .tracks import Track

try:
    import jax as _jax
    import jax.numpy as jnp
    from jax import jit
    from jax.random import normal as jnormal

    # Enable 64-bit in JAX to match NumPy float64 computations
    try:
        _jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
except ImportError:
    jnp = None
    jit = None
    jnormal = None
    if config["general"]["jax"]:
        raise ImportError("JAX not found! Install with: pip install jax jaxlib")


_log = logging.getLogger(__name__)


class Photon:
    """
    Cherenkov photon yield calculator.

    Calculates photon production for tracks and cascades, including
    wavelength-dependent yields and angular distributions.

    Attributes
    ----------
    _medium : str
        Medium name
    _n : float
        Refractive index
    _alpha : float
        Fine structure constant
    _charge : float
        Elementary charge
    _wavelengths : np.ndarray
        Wavelength grid in nm
    _angle_grid : np.ndarray
        Angular grid in degrees
    _zgrid : np.ndarray
        Depth grid in cm

    Examples
    --------
    >>> from fennel import Fennel
    >>> f = Fennel()
    >>> # Photon object used internally by Fennel

    Notes
    -----
    - Handles tracks (muons), EM cascades (e±, γ), and hadronic cascades
    - Supports both NumPy and JAX backends
    - Integrates all cascade components
    """

    def __init__(
        self,
        particle: Dict[int, Particle],
        track: Track,
        em_cascade: EM_Cascade,
        hadron_cascade: Hadron_Cascade,
    ) -> None:
        """
        Initialize the photon calculator.

        Parameters
        ----------
        particle : dict of {int: Particle}
            Dictionary of particle objects keyed by PDG ID
        track : Track
            Track calculator instance
        em_cascade : EM_Cascade
            EM cascade calculator instance
        hadron_cascade : Hadron_Cascade
            Hadron cascade calculator instance

        Raises
        ------
        ValueError
            If distribution type not implemented
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.debug("Constructing a photon object")
        self._medium = config["scenario"]["medium"]
        self._n = config["mediums"][self._medium]["refractive index"]
        self._alpha = config["advanced"]["fine structure"]
        self._charge = config["advanced"]["particle charge"]
        self._wavelengths = config["advanced"]["wavelengths"]
        self._angle_grid = config["advanced"]["angles"]
        self._zgrid = config["advanced"]["z grid"]
        self.__particles = particle
        self.__track = track
        self.__em_cascade = em_cascade
        self.__hadron_cascade = hadron_cascade
        self._rstate = config["runtime"]["random state"]
        self._deltaL = config["advanced"]["track length"]
        self._track_interactions = config["simulation"]["track interactions"]
        # Building the functions
        _log.info("Building the necessary functions")
        self._track_builder()
        self._em_cascade_builder()
        self._hadron_cascade_builder()
        _log.debug("Finished a photon object.")

    ###########################################################################
    # The Fetchers
    def _track_fetcher(
        self,
        energy: Union[float, np.ndarray],
        wavelengths: np.ndarray = config["advanced"]["wavelengths"],
        angle_grid: np.ndarray = config["advanced"]["angles"],
        n: float = config["mediums"][config["scenario"]["medium"]]["refractive index"],
        interaction: str = "total",
        function: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Callable, Callable]]:
        """
        Fetch Cherenkov photon yields for particle tracks.

        Calculates or returns functions for differential photon counts
        and angular distributions from particle tracks (currently muons).

        Parameters
        ----------
        energy : float or np.ndarray
            Particle energy(ies) in GeV
        wavelengths : np.ndarray, optional
            Wavelength grid in nm
        angle_grid : np.ndarray, optional
            Angular grid in degrees
        n : float, optional
            Refractive index of medium
        interaction : str, optional
            Interaction type: 'total', 'ionization', 'brems', 'pair'
        function : bool, optional
            If True, return callable functions; if False, return evaluated arrays

        Returns
        -------
        differential_counts : np.ndarray or Callable
            dN/dλ per cm of track length [photons/(nm·cm)]
            Shape: (n_wavelengths,) if evaluated
        angles : np.ndarray or Callable
            Angular distribution [photons/degree]

        Notes
        -----
        In JAX mode, functions only accept scalar inputs.
        """
        if function:
            _log.debug("Fetching track functions for " + interaction)
            return (
                self._track_functions_dic[interaction]["dcounts"],
                self._track_functions_dic[interaction]["angles"],
            )
        else:
            _log.debug("Fetching track values for " + interaction)
            if config["general"]["jax"]:
                # Use NumPy implementations for direct evaluations to match
                # NumPy backend results exactly (avoid tiny JAX numeric diffs)
                new_track = self.__track._additional_track_ratio_fetcher(
                    energy, interaction=interaction
                )
                dcounts_np = self._cherenkov_counts(
                    wavelengths, self._deltaL * (1.0 + new_track)
                )
                angles_np = self.__track._symmetric_angle_distro_fetcher(
                    angle_grid, n, energy
                )
                return np.asarray(dcounts_np, dtype=np.float64), np.asarray(
                    angles_np, dtype=np.float64
                )
            else:
                return (
                    self._track_functions_dic[interaction]["dcounts"](
                        energy, wavelengths
                    ),
                    self._track_functions_dic[interaction]["angles"](
                        angle_grid, n, energy
                    ),
                )

    def _em_cascade_fetcher(
        self,
        energy: Union[float, np.ndarray],
        particle: int,
        wavelengths: np.ndarray = config["advanced"]["wavelengths"],
        angle_grid: np.ndarray = config["advanced"]["angles"],
        n: float = config["mediums"][config["scenario"]["medium"]]["refractive index"],
        z_grid: np.ndarray = config["advanced"]["z grid"],
        function: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[Callable, Callable, Callable, Callable],
    ]:
        """
        Fetch Cherenkov photon yields for electromagnetic cascades.

        Parameters
        ----------
        energy : float or np.ndarray
            Cascade energy(ies) in GeV
        particle : int
            PDG particle code (11=e±, 22=γ)
        wavelengths : np.ndarray, optional
            Wavelength grid in nm
        angle_grid : np.ndarray, optional
            Angular grid in degrees
        n : float, optional
            Refractive index
        z_grid : np.ndarray, optional
            Depth grid in cm
        function : bool, optional
            If True, return callables; if False, return arrays

        Returns
        -------
        differential_counts : np.ndarray or Callable
            dN/dλ per cm [photons/(nm·cm)]
        differential_counts_sample : np.ndarray or Callable
            Sampled distribution
        long_profile : np.ndarray or Callable
            Longitudinal distribution [1/cm]
        angles : np.ndarray or Callable
            Angular distribution [photons/degree]
        """
        if function:
            _log.debug("Fetching em functions for pdg_id " + str(particle))
            return (
                self._em_cascade_function_dic[particle]["dcounts"],
                self._em_cascade_function_dic[particle]["dcounts sample"],
                self._em_cascade_function_dic[particle]["long distro"],
                self._em_cascade_function_dic[particle]["angle distro"],
            )
        # Fetching explicit values
        else:
            _log.debug("Fetching track values for " + str(particle))
            if config["general"]["jax"]:
                # Evaluate EM cascade using NumPy implementations to avoid
                # tiny numeric differences from JAX computations
                dcounts_np = (
                    self._em_cascade_function_dic[particle]["dcounts"](
                        energy, wavelengths
                    )
                    if not hasattr(
                        self._em_cascade_function_dic[particle]["dcounts"], "__call__"
                    )
                    else None
                )
                # The safer route: call the underlying EM_Cascade numpy methods
                tmp_track, _ = self.__em_cascade._track_lengths_fetcher(
                    energy, particle
                )
                dcounts_np = self._cherenkov_counts(wavelengths, tmp_track)
                dcounts_sample_np = self._cherenkov_counts(wavelengths, tmp_track)
                long_prof_np = self.__em_cascade._log_profile_func_fetcher(
                    energy, z_grid, particle
                )
                angles_np = self.__em_cascade._symmetric_angle_distro(
                    phi=angle_grid, n=n, name=particle
                )
                return (
                    np.asarray(dcounts_np, dtype=np.float64),
                    np.asarray(dcounts_sample_np, dtype=np.float64),
                    np.asarray(long_prof_np, dtype=np.float64),
                    np.asarray(angles_np, dtype=np.float64),
                )
            else:
                return (
                    self._em_cascade_function_dic[particle]["dcounts"](
                        energy, wavelengths
                    ),
                    self._em_cascade_function_dic[particle]["dcounts sample"](
                        energy, wavelengths
                    ),
                    self._em_cascade_function_dic[particle]["long distro"](
                        energy, z_grid
                    ),
                    self._em_cascade_function_dic[particle]["angle distro"](
                        angle_grid, n
                    ),
                )

    def _hadron_cascade_fetcher(
        self,
        energy: Union[float, np.ndarray],
        particle: int,
        wavelengths: np.ndarray = config["advanced"]["wavelengths"],
        angle_grid: np.ndarray = config["advanced"]["angles"],
        n: float = config["mediums"][config["scenario"]["medium"]]["refractive index"],
        z_grid: np.ndarray = config["advanced"]["z grid"],
        function: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[Callable, Callable, Callable, Callable, Callable, Callable],
    ]:
        """
        Fetch Cherenkov photon yields for hadronic cascades.

        Parameters
        ----------
        energy : float or np.ndarray
            Cascade energy(ies) in GeV
        particle : int
            PDG particle code (211=π±, 321=K±, 2212=p, 2112=n)
        wavelengths : np.ndarray, optional
            Wavelength grid in nm
        angle_grid : np.ndarray, optional
            Angular grid in degrees
        n : float, optional
            Refractive index
        z_grid : np.ndarray, optional
            Depth grid in cm
        function : bool, optional
            If True, return callables; if False, return arrays

        Returns
        -------
        differential_counts : np.ndarray or Callable
            dN/dλ per cm [photons/(nm·cm)]
        differential_counts_sample : np.ndarray or Callable
            Sampled distribution
        em_fraction_mean : np.ndarray or Callable
            Mean EM fraction
        em_fraction_sample : np.ndarray or Callable
            Sampled EM fraction
        long_profile : np.ndarray or Callable
            Longitudinal distribution [1/cm]
        angles : np.ndarray or Callable
            Angular distribution [photons/degree]
        """
        if function:
            _log.debug("Fetching em functions for pdg_id " + str(particle))
            return (
                self._hadron_cascade_function_dic[particle]["dcounts"],
                self._hadron_cascade_function_dic[particle]["dcounts sample"],
                self._hadron_cascade_function_dic[particle]["em fraction mean"],
                self._hadron_cascade_function_dic[particle]["em fraction sample"],
                self._hadron_cascade_function_dic[particle]["long distro"],
                self._hadron_cascade_function_dic[particle]["angle distro"],
            )
        # Fetching explicit values
        else:
            _log.debug("Fetching track values for " + str(particle))
            if config["general"]["jax"]:
                # Use NumPy hadron-cascade implementations for direct evaluations
                tmp_track, _ = self.__hadron_cascade.track_lengths(energy, particle)
                dcounts_np = self._cherenkov_counts(wavelengths, tmp_track)
                dcounts_sample_np = self._cherenkov_counts(wavelengths, tmp_track)
                em_frac_mean = float(
                    self.__hadron_cascade.em_fraction(energy, particle)[0]
                )
                em_frac_sample = float(
                    self.__hadron_cascade.em_fraction(energy, particle)[0]
                )
                long_prof_np = self.__hadron_cascade.long_profile(
                    energy, z_grid, particle
                )
                angles_np = self.__hadron_cascade.cherenkov_angle_distro(
                    energy, angle_grid, n, particle
                )
                return (
                    np.asarray(dcounts_np, dtype=np.float64),
                    np.asarray(dcounts_sample_np, dtype=np.float64),
                    em_frac_mean,
                    em_frac_sample,
                    np.asarray(long_prof_np, dtype=np.float64),
                    np.asarray(angles_np, dtype=np.float64),
                )
            else:
                return (
                    self._hadron_cascade_function_dic[particle]["dcounts"](
                        energy, wavelengths
                    ),
                    self._hadron_cascade_function_dic[particle]["dcounts sample"](
                        energy, wavelengths
                    ),
                    self._hadron_cascade_function_dic[particle]["em fraction mean"](
                        energy
                    ),
                    self._hadron_cascade_function_dic[particle]["em fraction sample"](
                        energy
                    ),
                    self._hadron_cascade_function_dic[particle]["long distro"](
                        energy, z_grid
                    ),
                    self._hadron_cascade_function_dic[particle]["angle distro"](
                        energy, angle_grid, n
                    ),
                )

    ###########################################################################
    # The Builders

    def _track_builder(self):
        """Builder function for the track functions.

        Parameters
        ----------
        interaction : str
            Optional: The interaction(s) which should produce the light

        Returns
        -------
        None
        """
        # Looping over the interaction types
        _log.info("Building track functions")
        self._track_functions_dic = {}
        for interaction in self._track_interactions:
            # Photon count function and angle function
            self._track_functions_dic[interaction] = {}

            # Building the counts functionn
            def track_mean(energy, interaction=interaction):
                """Fetcher function for a specific particle
                and energy. This is for tracks

                Parameters
                ----------
                energy : float
                    The energy of the particle
                interaction : str
                    The interaction type

                Returns
                -------
                counts : float
                    The photon counts
                """
                tmp_track_frac = self.__track.additional_track_ratio(
                    energy, interaction=interaction
                )
                new_track = self._deltaL * (1.0 + tmp_track_frac)
                return new_track

            if config["general"]["jax"]:
                _log.debug("Constructing Jax function for " + interaction)

                def counts_mean(energy, wavelengths, interaction=interaction):
                    """Calculates the differential photon counts.
                    Jax implemenation
                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float/np.array
                        The wavelength(s) of interest
                    interaction : str
                        The interaction type

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, interaction=interaction)
                    return self._cherenkov_counts_jax(wavelengths, new_track)

                # jitting
                counts = jit(counts_mean, static_argnames=["interaction"])
                angles = jit(self.__track.cherenkov_angle_distro)
            else:
                _log.debug("Constructing Jax function for " + interaction)

                def counts_mean(energy, wavelengths, interaction=interaction):
                    """Calculates the differential photon counts.

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float/np.array
                        The wavelength(s) of interest
                    interaction : str
                        The interaction type

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, interaction=interaction)
                    return self._cherenkov_counts(wavelengths, new_track)

                # No jitting here
                counts = counts_mean
                angles = self.__track.cherenkov_angle_distro
            self._track_functions_dic[interaction]["dcounts"] = counts
            self._track_functions_dic[interaction]["angles"] = angles

    def _em_cascade_builder(self):
        """Builder function for a the cascade functions. This is for
        em cascades.

        Parameters
        ----------

        Returns
        -------
        None
        """
        _log.debug("Building the em cascade functions")
        self._em_cascade_function_dic = {}
        for particle_id in config["simulation"]["em particles"]:
            name = particle_id
            self._em_cascade_function_dic[particle_id] = {}

            def track_mean(energy, name=name):
                """Fetcher function for a specific particle and energy.
                This is for em cascades and their photon counts

                Parameters
                ----------
                energy : float
                    The energy of the particle
                name : int
                    The name of the particle of interest

                Returns
                -------
                tmp_track : float
                    The track length
                """
                tmp_track, _ = self.__em_cascade.track_lengths(energy, name)
                return tmp_track

            def track_sampler(energy, name=name):
                """Fetcher function for a specific particle and energy.
                This samples the distribution

                Parameters
                ----------
                energy : float
                    The energy of the particle
                name : int
                    The name of the particle of interest

                Returns
                -------
                tmp_track_sample : float
                    The sampled photon counts
                """
                tmp_track, tmp_track_sd = self.__em_cascade.track_lengths(energy, name)
                if config["general"]["jax"]:
                    tmp_track_sample = tmp_track + tmp_track_sd * jnormal(self._rstate)
                else:
                    tmp_track_sample = self._rstate.normal(tmp_track, tmp_track_sd)
                return tmp_track_sample

            def long_profile(energy, z_grid, name=name):
                """The longitudinal profile of the em cascade

                Parameters
                ----------
                energy : float
                    The energy of the particle
                z_grid : float/np.array
                    The grid to evaluate the distribution in in cm
                name : int
                    The name of the particle of interest

                Returns
                -------
                long_distro : float/np.array
                    The resulting longitudinal distribution
                """
                return self.__em_cascade.long_profile(energy, z_grid, name)

            def angle_distro(
                angles, n=config["mediums"][self._medium]["refractive index"], name=name
            ):
                """The angle distribution of the cherenkov photons for
                the em cascade

                Parameters
                ----------
                angles : float/np.array
                    The angles of interest
                n : float
                    Optional: The refractive index of the material
                name : int
                    Optional: The name of the particle of interest

                Returns
                -------
                angle_distro : float/np.array
                    The resulting longitudinal distribution
                """
                return self.__em_cascade.cherenkov_angle_distro(angles, n, name)

            # Storing the functions
            if config["general"]["jax"]:
                _log.debug("Constructing Jax function for pdg_id " + str(name))

                def counts_mean(energy, wavelengths, name=name):
                    """Calculates the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : float
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, name=name)
                    return self._cherenkov_counts_jax(wavelengths, new_track)

                def counts_sampler(energy, wavelengths, name=name):
                    """Calculates a sample of the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : float
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_sampler(energy, name=name)
                    return self._cherenkov_counts_jax(wavelengths, new_track)

                # Jit the jax functions
                counts = jit(counts_mean, static_argnames=["name"])
                counts_sample = jit(counts_sampler, static_argnames=["name"])
                long = jit(long_profile, static_argnames=["name"])
                angles = jit(angle_distro, static_argnames=["name"])
            else:
                _log.debug("Constructing numpy function for pdg_id " + str(name))

                def counts_mean(energy, wavelengths, name=name):
                    """Calculates the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: np.array
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, name=name)
                    return self._cherenkov_counts(wavelengths, new_track)

                def counts_sampler(energy, wavelengths, name=name):
                    """Calculates a sample of the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: np.array
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_sampler(energy, name=name)
                    return self._cherenkov_counts(wavelengths, new_track)

                # Don't jist the numpy functions
                counts = counts_mean
                counts_sample = counts_sampler
                long = long_profile
                angles = angle_distro
            # Storing
            self._em_cascade_function_dic[particle_id] = {
                "dcounts": counts,
                "dcounts sample": counts_sample,
                "long distro": long,
                "angle distro": angles,
            }

    def _hadron_cascade_builder(self):
        """Builder function for a hadronic cascades.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _log.debug("Building the hadron cascade functions")
        self._hadron_cascade_function_dic = {}
        for particle_id in config["simulation"]["hadron particles"]:
            name = particle_id
            self._hadron_cascade_function_dic[particle_id] = {}

            def track_mean(energy, name=name):
                """Fetcher function for a specific particle and energy.
                This is for hadron cascades and their photon counts

                Parameters
                ----------
                energy : float
                    The energy of the particle
                name : int
                    The name of the particle of interest

                Returns
                -------
                tmp_track : float
                    The track length
                """
                tmp_track, _ = self.__hadron_cascade.track_lengths(energy, name)
                return tmp_track

            def track_sampler(energy, name=name):
                """Fetcher function for a specific particle and energy.
                This samples the distribution

                Parameters
                ----------
                energy : float
                    The energy of the particle
                name : int
                    The name of the particle of interest

                Returns
                -------
                tmp_track_sample : float
                    The sampled photon counts
                """
                tmp_track, tmp_track_sd = self.__hadron_cascade.track_lengths(
                    energy, name
                )
                if config["general"]["jax"]:
                    tmp_track_sample = tmp_track + tmp_track_sd * jnormal(self._rstate)
                else:
                    tmp_track_sample = self._rstate.normal(tmp_track, tmp_track_sd)
                return tmp_track_sample

            def em_fraction_mean(energy, name=name):
                """The em fraction mean of the hadron cascade

                Parameters
                ----------
                energy : float
                    The energy of the particle
                name : int
                    The name of the particle of interest

                Returns
                -------
                em_frac_mean : float/np.array
                    The resulting em fraction
                """
                em_frac_mean, _ = self.__hadron_cascade.em_fraction(energy, name)
                return em_frac_mean

            def em_fraction_sampler(energy, name=name):
                """The em fraction sample of the hadron cascade

                Parameters
                ----------
                energy : float
                    The energy of the particle
                name : int
                    The name of the particle of interest

                Returns
                -------
                em_frac_sample : float/np.array
                    The resulting em fraction
                """
                em_frac_mean, em_frac_std = self.__hadron_cascade.em_fraction(
                    energy, name
                )
                if config["general"]["jax"]:
                    em_frac_sample = em_frac_mean + em_frac_std * jnormal(self._rstate)
                else:
                    em_frac_sample = self._rstate.normal(em_frac_mean, em_frac_std)
                return em_frac_sample

            def long_profile(energy, z_grid, name=name):
                """The longitudinal profile of the hadron cascade

                Parameters
                ----------
                energy : float
                    The energy of the particle
                z_grid : float/np.array
                    The grid to evaluate the distribution in in cm
                name : int
                    The name of the particle of interest

                Returns
                -------
                long_distro : float/np.array
                    The resulting longitudinal distribution
                """
                return self.__hadron_cascade.long_profile(energy, z_grid, name)

            def angle_distro(
                energy,
                angles,
                n=config["mediums"][self._medium]["refractive index"],
                name=name,
            ):
                """The angle distribution of the cherenkov photons for
                the hadron cascade

                Parameters
                ----------
                angles : float/np.array
                    The angles of interest
                n : float
                    Optional: The refractive index of the material
                name : int
                    Optional: The name of the particle of interest

                Returns
                -------
                angle_distro : float/np.array
                    The resulting longitudinal distribution
                """
                return self.__hadron_cascade.cherenkov_angle_distro(
                    energy, angles, n, name
                )

            # Storing the functions
            if config["general"]["jax"]:
                _log.debug("Constructing Jax function for pdg_id " + str(name))

                def counts_mean(energy, wavelengths, name=name):
                    """Calculates the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : float
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, name=name)
                    return self._cherenkov_counts_jax(wavelengths, new_track)

                def counts_sampler(energy, wavelengths, name=name):
                    """Calculates a sample of the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : float
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_sampler(energy, name=name)
                    return self._cherenkov_counts_jax(wavelengths, new_track)

                # Jit the jax functions
                counts = jit(counts_mean, static_argnames=["name"])
                counts_sample = jit(counts_sampler, static_argnames=["name"])
                em_frac_mean = jit(em_fraction_mean, static_argnames=["name"])
                em_frac_sample = jit(em_fraction_sampler, static_argnames=["name"])
                long = jit(long_profile, static_argnames=["name"])
                angles = jit(angle_distro, static_argnames=["name"])
            else:
                _log.debug("Constructing numpy function for pdg_id " + str(name))

                def counts_mean(energy, wavelengths, name=name):
                    """Calculates the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: np.array
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, name=name)
                    return self._cherenkov_counts(wavelengths, new_track)

                def counts_sampler(energy, wavelengths, name=name):
                    """Calculates a sample of the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: np.array
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_sampler(energy, name=name)
                    return self._cherenkov_counts(wavelengths, new_track)

                # Don't jist the numpy functions
                counts = counts_mean
                counts_sample = counts_sampler
                em_frac_mean = em_fraction_mean
                em_frac_sample = em_fraction_sampler
                long = long_profile
                angles = angle_distro
            # Storing
            self._hadron_cascade_function_dic[particle_id] = {
                "dcounts": counts,
                "dcounts sample": counts_sample,
                "em fraction mean": em_frac_mean,
                "em fraction sample": em_frac_sample,
                "long distro": long,
                "angle distro": angles,
            }

    def _cherenkov_counts(self, wavelengths: np.array, track_length: float) -> np.array:
        """Calculates the differential number of photons for the given
        wavelengths and track-lengths assuming a constant velocity with beta=1.

        Parameters
        ----------
        wavelengths : np.array
            The wavelengths of interest
        track_lengths : float
            The track lengths of interest in cm

        Returns
        -------
        counts : np.array
            A array filled witht the produced photons.
        """
        prefac = (
            2.0 * np.pi * self._alpha * self._charge**2.0 / (1.0 - 1.0 / self._n**2.0)
        )
        # 1e-7 due to the conversion from nm to cm
        diff_counts = np.array(
            [
                prefac / (lambd * 1e-9) ** 2.0 * track_length * 1e-2
                for lambd in wavelengths
            ]
        )
        return diff_counts * 1e-9

    def _cherenkov_counts_jax(self, wavelengths: float, track_lengths: float) -> float:
        """Calculates the differential number of photons for the given
        wavelengths and track-lengths assuming a constant velocity with beta=1.

        Parameters
        ----------
        wavelengths : float
            The wavelengths of interest
        track_lengths : float
            The track lengths of interest in cm

        Returns
        -------
        counts : float
            The counts (differential)
        """
        prefac = (
            2.0 * jnp.pi * self._alpha * self._charge**2.0 / (1.0 - 1.0 / self._n**2.0)
        )
        # 1e-7 due to the conversion from nm to cm
        diff_counts = prefac / (wavelengths * 1e-9) ** 2.0 * track_lengths * 1e-2
        return diff_counts * 1e-9
