# -*- coding: utf-8 -*-
"""
Main interface to the Fennel light yield model.

This module implements the Aachen parametrization for calculating Cherenkov light
yields from particle interactions in transparent media. The parametrization is based
on Leif Raedel's Master thesis:
https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaaapwhjz
"""

# Imports
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import yaml

# Package modules
from .config import config
from .definition_generator import Definitions_Generator
from .em_cascades import EM_Cascade
from .hadron_cascades import Hadron_Cascade
from .particle import Particle
from .photons import Photon
from .results import EMYieldResult, HadronYieldResult, TrackYieldResult
from .tracks import Track
from .validation import (
    suggest_particle_type,
    validate_energy,
    validate_interaction,
    validate_particle_pdg,
    validate_refractive_index,
    validate_wavelengths,
)

try:
    from jax.random import PRNGKey
except ImportError:
    PRNGKey = None
    if config["general"]["jax"]:
        raise ImportError("JAX not found! Install with: pip install jax jaxlib")


_log = logging.getLogger("fennel")


class Fennel:
    """
    Main interface for light yield calculations using the Aachen parametrization.

    This class provides methods for calculating Cherenkov light yields from
    various particle types (tracks, electromagnetic cascades, hadronic cascades)
    in transparent media.

    Parameters
    ----------
    userconfig : dict or str or Path, optional
        User configuration as either a dictionary or path to YAML file.
        If None, uses the default configuration from config module.

    Attributes
    ----------
    _particles : Dict[int, Particle]
        Dictionary mapping PDG IDs to Particle objects
    _track : Track
        Track light yield calculator
    _em_cascade : EM_Cascade
        Electromagnetic cascade light yield calculator
    _hadron_cascade : Hadron_Cascade
        Hadronic cascade light yield calculator
    _photon : Photon
        Photon propagation calculator
    _dg : Definitions_Generator
        Generator for storing calculation definitions

    Examples
    --------
    Basic usage with default configuration:

    >>> from fennel import Fennel
    >>> fennel = Fennel()
    >>> energy = 100.0  # GeV
    >>> wavelengths = np.linspace(300, 600, 100)  # nm
    >>> dcounts, angles = fennel.track_yields(energy, wavelengths=wavelengths)
    >>> fennel.close()

    Using custom configuration:

    >>> from fennel import Fennel, config
    >>> config["general"]["random state seed"] = 42
    >>> config["general"]["jax"] = False
    >>> fennel = Fennel()
    >>> dcounts, _, _, _ = fennel.em_yields(energy=1000.0, particle=11)
    >>> fennel.close()

    Loading configuration from file:

    >>> fennel = Fennel("my_config.yaml")

    Notes
    -----
    - Always call `close()` when finished to properly cleanup and save logs
    - Enable JAX for GPU acceleration of calculations
    - Set random seed for reproducible results
    """

    def __init__(
        self, userconfig: Optional[Union[Dict[str, Any], str, Path]] = None
    ) -> None:
        """
        Initialize the Fennel light yield calculator.

        Parameters
        ----------
        userconfig : dict or str or Path, optional
            User configuration. Can be:
            - dict: Configuration dictionary to merge with defaults
            - str or Path: Path to YAML configuration file
            - None: Use default configuration

        Raises
        ------
        ImportError
            If JAX is enabled in config but not installed
        FileNotFoundError
            If configuration file path doesn't exist
        """
        # Inputs
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        # Create RandomState
        if config["general"]["random state seed"] is None:
            _log.warning("No random state seed given, constructing new state")
            if config["general"]["jax"]:
                rstate = PRNGKey(1337)
            else:
                rstate = np.random.RandomState()
        else:
            if config["general"]["jax"]:
                rstate = PRNGKey(config["general"]["random state seed"])
            else:
                rstate = np.random.RandomState(config["general"]["random state seed"])
        config["runtime"] = {"random state": rstate}

        # Logger
        # Logging formatter
        fmt = "%(levelname)s: %(message)s"
        fmt_with_name = "[%(name)s] " + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        # creating file handler with debug messages
        if config["general"]["enable logging"]:
            fh = logging.FileHandler(config["general"]["log file handler"], mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter_with_name)
            _log.addHandler(fh)
        else:
            _log.disabled = True
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(config["general"]["debug level"])
        # add class name to ch only when debugging
        if config["general"]["debug level"] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)
        _log.addHandler(ch)
        _log.setLevel(logging.DEBUG)
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        _log.info("Welcome to Fennel!")
        _log.info("This package will help you model light yields")
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        _log.info("Creating particles...")
        self._particles = {}
        for particle_id in config["pdg id"].keys():
            # Particle creation
            self._particles[particle_id] = Particle(particle_id)
        _log.info("Creation finished")
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        _log.info("Creating a track...")
        # Track creation
        self._track = Track()
        _log.info("Creation finished")
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        _log.info("Creating an em cascade...")
        # EM cascade creation
        self._em_cascade = EM_Cascade()
        _log.info("Creation finished")
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        _log.info("Creating a hadron cascade...")
        # Hadron cascade creation
        self._hadron_cascade = Hadron_Cascade()
        _log.info("Creation finished")
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        _log.info("Creating a photon...")
        # Hadron cascade creation
        self._photon = Photon(
            self._particles, self._track, self._em_cascade, self._hadron_cascade
        )
        # Creating the definitions storer
        self._dg = Definitions_Generator(
            self._track, self._em_cascade, self._hadron_cascade
        )
        _log.info("Creation finished")
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")

    def close(self) -> None:
        """
        Clean up and finalize the Fennel session.

        Saves the configuration to file (if logging is enabled) and closes
        all logging handlers. Always call this method when finished with
        calculations to ensure proper cleanup.

        Examples
        --------
        >>> fennel = Fennel()
        >>> # ... perform calculations ...
        >>> fennel.close()

        Notes
        -----
        This method will:
        - Dump the current configuration to the location specified in config
        - Display a farewell message in the logs
        - Shut down all logging handlers
        """
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        # A new simulation
        if config["general"]["enable logging"]:
            _log.debug(
                "Dumping run settings into %s",
                config["general"]["config location"],
            )
            with open(config["general"]["config location"], "w") as f:
                yaml.dump(config, f)
            _log.debug("Finished dump")
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        _log.info("Have a great day and until next time!")
        _log.info("                  @*****&@         @@.                    ")
        _log.info("           @@@((@ @*******@     @%((((@@((@               ")
        _log.info("         @(((((((@(@*******@@@((@(((((((((@*              ")
        _log.info("        @((((((@(((@*******@(((@%(((((&%(((@              ")
        _log.info("         #@((@((&@((&*******@((@(((((@#(((@               ")
        _log.info("        @@*****@((#@@********@@(((((@(((@/**@@            ")
        _log.info("        @********@((@@*************@@@#@******&%          ")
        _log.info("@@  @ @&(@(***,*******,*******,*******,*******,@      .@@*")
        _log.info(" @   @@((((@*************.*********...*******@(@      ./  ")
        _log.info(" @     %(((#@**********...*,.. ........**,**(@(#*     @   ")
        _log.info("  @      @@,.*.***,@@@.......*,...@@@..*,.....@@    &@    ")
        _log.info("   @(    @....,,...@@......... ,*.@%...*......,%  @@      ")
        _log.info("       @@@.....,***...............,****.......*@,         ")
        _log.info("        .........*............ ........****..*.@          ")
        _log.info("         #*........*...&       @...............@          ")
        _log.info("         @ *.. ... ..,,..@@@@. ... ... ... ...&           ")
        _log.info("          @.,*...........*,...................@           ")
        _log.info("            @.**.............*,..............@            ")
        _log.info("             .@.**...............**.........@             ")
        _log.info("           @@@%   &@@,........ .......**#@    @@          ")
        _log.info("        @,                 (@@@@@@@@(             @       ")
        _log.info("                                                %@*       ")
        _log.info("---------------------------------------------------")
        _log.info("---------------------------------------------------")
        # Closing log
        logging.shutdown()

    def auto_yields(
        self,
        energy,
        particle: int,
        interaction="total",
        wavelengths=config["advanced"]["wavelengths"],
        angle_grid=config["advanced"]["angles"],
        n=config["mediums"][config["scenario"]["medium"]]["refractive index"],
        z_grid=config["advanced"]["z grid"],
        function=False,
    ):
        """Auto fetcher function for a given particle and energy. This will
        fetch/evaluate the functions corresponding to the given particle.
        Some of the output will be none depending on the constructed object

        Parameters
        ----------
        energy : float
            The energy(ies) of the particle in GeV
        particle : int
            The pdg id of the particle of interest
        wavelengths : np.array
            Optional: The desired wavelengths
        interaction : str
            Optional: The interaction which should produce the light.
            This is used during track construction.
        angle_grid : np.array
            Optional: The desired angles in degress
        n : float
            Optional: The refractive index of the medium.
        z_grid : np.array
            Optional: The grid in cm for the long. distributions.
            Used when modeling cascades.
        function : bool
            Optional: returns the functional form instead of the evaluation

        Returns
        -------
        differential_counts : function/float/np.array
            dN/dlambda The differential photon counts per track length (in cm).
            The shape of the array is (len(wavelengths), len(deltaL)).
        differential_counts_sample : float/np.array
            A sample of the differential counts distribution. Same shape as
            the differential counts
        em_fraction_mean : float/np.array
            The fraction of em particles
        em_fraction_sample : float/np.array
            A sample of the em_fraction
        long_profile : function/float/np.array
            The distribution along the shower axis for cm
        angles : function/float/np.array
            The angular distribution in degrees
        """
        if particle in config["simulation"]["track particles"]:
            _log.debug("Fetching/evaluating track functions for " + str(particle))
            dcounts, angles = self.track_yields(
                energy,
                wavelengths=wavelengths,
                angle_grid=angle_grid,
                n=n,
                interaction=interaction,
                function=function,
            )
            # Unfilled variables
            dcounts_s = None
            em_frac = None
            em_frac_s = None
            long = None
        elif particle in config["simulation"]["em particles"]:
            _log.debug("Fetching/evaluating em functions for " + str(particle))
            dcounts, dcounts_s, long, angles = self.em_yields(
                energy,
                particle,
                wavelengths=wavelengths,
                angle_grid=angle_grid,
                n=n,
                z_grid=z_grid,
                function=function,
            )
            # Unfilled variables
            em_frac = None
            em_frac_s = None
        elif particle in config["simulation"]["hadron particles"]:
            _log.debug("Fetching/evaluating hadron functions for " + str(particle))
            dcounts, dcounts_s, em_frac, em_frac_s, long, angles = self.hadron_yields(
                energy,
                particle,
                wavelengths=wavelengths,
                angle_grid=angle_grid,
                n=n,
                z_grid=z_grid,
                function=function,
            )
        else:
            raise ValueError(
                "Track/cascade object corresponding to "
                + str(particle)
                + " is unknown. Please contact "
                + "the authors if there is a need for this species"
            )
        return dcounts, dcounts_s, em_frac, em_frac_s, long, angles

    def track_yields(
        self,
        energy: float,
        wavelengths: Optional[np.ndarray] = None,
        angle_grid: Optional[np.ndarray] = None,
        n: Optional[float] = None,
        interaction: str = "total",
        function: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Callable, Callable]]:
        """
        Calculate Cherenkov light yields from charged particle tracks (muons).

        Computes the differential photon counts as a function of wavelength
        and the angular distribution of emitted Cherenkov light for a charged
        particle track (currently muons only).

        Parameters
        ----------
        energy : float
            Particle energy in GeV. Must be positive.
        wavelengths : np.ndarray, optional
            Wavelengths at which to calculate yields in nm.
            If None, uses config["advanced"]["wavelengths"].
            Shape: (n_wavelengths,)
        angle_grid : np.ndarray, optional
            Emission angles in degrees for angular distribution.
            If None, uses config["advanced"]["angles"].
            Shape: (n_angles,)
        n : float, optional
            Refractive index of the medium.
            If None, uses value from config for current medium.
            Must be > 1 for Cherenkov emission.
        interaction : {'total', 'ionization', 'brems', 'pair', 'nuclear'}, optional
            Energy loss mechanism to consider:
            - 'total': All interactions combined (default)
            - 'ionization': Ionization losses only
            - 'brems': Bremsstrahlung only
            - 'pair': Pair production only
            - 'nuclear': Nuclear interactions only
        function : bool, optional
            If True, returns callable functions instead of evaluated arrays.
            In JAX mode, these functions only accept scalar inputs.
            Default is False.

        Returns
        -------
        differential_counts : np.ndarray or Callable
            If function=False: Array of differential photon counts dN/dλ
            per cm of track length. Shape: (n_wavelengths,)
            If function=True: Callable with signature (energy, wavelength) -> float
        angles : np.ndarray or Callable
            If function=False: Angular distribution of emitted light.
            Shape: (n_angles,)
            If function=True: Callable with signature (angle, n, energy) -> float

        Examples
        --------
        Calculate light yield for 100 GeV muon:

        >>> fennel = Fennel()
        >>> wavelengths = np.linspace(300, 600, 100)
        >>> energy = 100.0  # GeV
        >>> dcounts, angles = fennel.track_yields(energy, wavelengths=wavelengths)
        >>> total_photons_per_cm = integrate_trapezoid(dcounts, wavelengths)

        Get functional form for later evaluation:

        >>> dcounts_func, angles_func = fennel.track_yields(
        ...     energy, function=True
        ... )
        >>> yield_at_400nm = dcounts_func(energy, 400.0)

        Calculate only bremsstrahlung contribution:

        >>> dcounts_brems, _ = fennel.track_yields(
        ...     energy, interaction='brems'
        ... )

        Notes
        -----
        - Currently only supports muons (PDG ID 13, -13)
        - In JAX mode with function=True, returned functions are JIT-compiled
        - The angular distribution is normalized to integrate to 1
        - Wavelengths should be in the optical/UV range (typically 300-600 nm)
        """
        if wavelengths is None:
            wavelengths = config["advanced"]["wavelengths"]
        if angle_grid is None:
            angle_grid = config["advanced"]["angles"]
        if n is None:
            n = config["mediums"][config["scenario"]["medium"]]["refractive index"]
        return self._photon._track_fetcher(
            energy, wavelengths, angle_grid, n, interaction, function
        )

    def em_yields(
        self,
        energy: float,
        particle: int,
        wavelengths: Optional[np.ndarray] = None,
        angle_grid: Optional[np.ndarray] = None,
        n: Optional[float] = None,
        z_grid: Optional[np.ndarray] = None,
        function: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[Callable, Callable, Callable, Callable],
    ]:
        """
        Calculate Cherenkov light yields from electromagnetic cascades.

        Computes photon yields from electromagnetic showers initiated by
        electrons, positrons, or photons. Returns the spectral distribution,
        longitudinal profile, and angular distribution of emitted light.

        Parameters
        ----------
        energy : float
            Initial particle energy in GeV. Must be positive.
        particle : int
            PDG ID of the particle:
            - 11: electron (e-)
            - -11: positron (e+)
            - 22: photon (γ)
        wavelengths : np.ndarray, optional
            Wavelengths for spectral calculation in nm.
            If None, uses config["advanced"]["wavelengths"].
            Shape: (n_wavelengths,)
        angle_grid : np.ndarray, optional
            Emission angles in degrees.
            If None, uses config["advanced"]["angles"].
            Shape: (n_angles,)
        n : float, optional
            Refractive index of the medium.
            If None, uses value from config.
            Must be > 1 for Cherenkov emission.
        z_grid : np.ndarray, optional
            Distance grid for longitudinal profile in cm.
            If None, uses config["advanced"]["z grid"].
            Shape: (n_distances,)
        function : bool, optional
            If True, returns callable functions.
            Default is False.

        Returns
        -------
        differential_counts : np.ndarray or Callable
            Differential photon counts dN/dλ per cascade.
            Shape: (n_wavelengths,) if function=False
        differential_counts_sample : np.ndarray or Callable
            Sampled version of differential counts for stochastic modeling.
            Same shape as differential_counts.
        long_profile : np.ndarray or Callable
            Longitudinal shower development profile.
            Shape: (n_distances,) if function=False
        angles : np.ndarray or Callable
            Angular distribution of emitted Cherenkov light.
            Shape: (n_angles,) if function=False

        Examples
        --------
        Calculate light yield from 1 TeV electron:

        >>> fennel = Fennel()
        >>> energy = 1000.0  # GeV
        >>> wavelengths = np.linspace(300, 600, 100)
        >>> dcounts, dcounts_sample, long_prof, angles = fennel.em_yields(
        ...     energy, particle=11, wavelengths=wavelengths
        ... )
        >>> total_photons = integrate_trapezoid(dcounts, wavelengths)

        Compare electron and positron yields:

        >>> e_minus = fennel.em_yields(100.0, particle=11)
        >>> e_plus = fennel.em_yields(100.0, particle=-11)

        Get functional form:

        >>> dcounts_func, _, long_func, _ = fennel.em_yields(
        ...     energy, particle=22, function=True
        ... )

        Notes
        -----
        - Electromagnetic cascades develop through pair production and bremsstrahlung
        - The longitudinal profile shows shower development along the cascade axis
        - Electrons and positrons produce nearly identical yields
        - Photons initiate cascades through pair production
        """
        if wavelengths is None:
            wavelengths = config["advanced"]["wavelengths"]
        if angle_grid is None:
            angle_grid = config["advanced"]["angles"]
        if n is None:
            n = config["mediums"][config["scenario"]["medium"]]["refractive index"]
        if z_grid is None:
            z_grid = config["advanced"]["z grid"]
        return self._photon._em_cascade_fetcher(
            energy, particle, wavelengths, angle_grid, n, z_grid, function
        )

    def hadron_yields(
        self,
        energy: float,
        particle: int,
        wavelengths: Optional[np.ndarray] = None,
        angle_grid: Optional[np.ndarray] = None,
        n: Optional[float] = None,
        z_grid: Optional[np.ndarray] = None,
        function: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray],
        Tuple[Callable, Callable, Callable, Callable, Callable, Callable],
    ]:
        """
        Calculate Cherenkov light yields from hadronic cascades.

        Computes photon yields from hadronic showers initiated by pions,
        kaons, protons, or neutrons. Returns spectral and spatial distributions
        along with the electromagnetic fraction of the cascade.

        Parameters
        ----------
        energy : float
            Initial hadron energy in GeV. Must be positive.
        particle : int
            PDG ID of the hadron:
            - 211: π+ (positive pion)
            - -211: π- (negative pion)
            - 130: K_L0 (long-lived neutral kaon)
            - 2212: p (proton)
            - -2212: p̄ (antiproton)
            - 2112: n (neutron)
        wavelengths : np.ndarray, optional
            Wavelengths for spectral calculation in nm.
            If None, uses config["advanced"]["wavelengths"].
            Shape: (n_wavelengths,)
        angle_grid : np.ndarray, optional
            Emission angles in degrees.
            If None, uses config["advanced"]["angles"].
            Shape: (n_angles,)
        n : float, optional
            Refractive index of the medium.
            If None, uses value from config.
            Must be > 1 for Cherenkov emission.
        z_grid : np.ndarray, optional
            Distance grid for longitudinal profile in cm.
            If None, uses config["advanced"]["z grid"].
            Shape: (n_distances,)
        function : bool, optional
            If True, returns callable functions.
            Default is False.

        Returns
        -------
        differential_counts : np.ndarray or Callable
            Differential photon counts dN/dλ per cascade.
            Shape: (n_wavelengths,) if function=False
        differential_counts_sample : np.ndarray or Callable
            Sampled version for stochastic modeling.
            Same shape as differential_counts.
        em_fraction_mean : float or Callable
            Mean electromagnetic fraction of the cascade.
            Typically 0.5-0.9 depending on energy and particle type.
        em_fraction_sample : float or Callable
            Sampled electromagnetic fraction for stochastic modeling.
        long_profile : np.ndarray or Callable
            Longitudinal shower development profile.
            Shape: (n_distances,) if function=False
        angles : np.ndarray or Callable
            Angular distribution of emitted Cherenkov light.
            Shape: (n_angles,) if function=False

        Examples
        --------
        Calculate yield from 100 GeV positive pion:

        >>> fennel = Fennel()
        >>> energy = 100.0  # GeV
        >>> dcounts, dcounts_s, em_frac, em_frac_s, long_prof, angles = \
        ...     fennel.hadron_yields(energy, particle=211)
        >>> print(f"EM fraction: {em_frac:.2f}")
        EM fraction: 0.74

        Compare different hadrons:

        >>> pion_yields = fennel.hadron_yields(1000.0, particle=211)
        >>> proton_yields = fennel.hadron_yields(1000.0, particle=2212)
        >>> kaon_yields = fennel.hadron_yields(1000.0, particle=130)

        Notes
        -----
        - Hadronic cascades have both electromagnetic and hadronic components
        - The EM fraction increases with energy
        - Protons tend to have lower EM fractions than pions at same energy
        - The longitudinal profile is typically longer than EM cascades
        - Particle/antiparticle pairs may have slightly different yields
        """
        if wavelengths is None:
            wavelengths = config["advanced"]["wavelengths"]
        if angle_grid is None:
            angle_grid = config["advanced"]["angles"]
        if n is None:
            n = config["mediums"][config["scenario"]["medium"]]["refractive index"]
        if z_grid is None:
            z_grid = config["advanced"]["z grid"]
        return self._photon._hadron_cascade_fetcher(
            energy, particle, wavelengths, angle_grid, n, z_grid, function
        )

    # =========================================================================
    # New v2.0 API Methods with Result Containers and Better Validation
    # =========================================================================

    def track_yields_v2(
        self,
        energy: float,
        wavelengths: Optional[np.ndarray] = None,
        angle_grid: Optional[np.ndarray] = None,
        n: Optional[float] = None,
        interaction: str = "total",
        function: bool = False,
    ) -> TrackYieldResult:
        """
        Calculate track light yields with enhanced API (v2.0).

        This is an improved version of track_yields() that returns a structured
        result container and includes comprehensive input validation with
        helpful error messages.

        Parameters
        ----------
        energy : float
            Particle energy in GeV. Must be positive.
        wavelengths : np.ndarray, optional
            Wavelength grid in nm. If None, uses config default.
        angle_grid : np.ndarray, optional
            Angular grid in radians. If None, uses config default.
        n : float, optional
            Refractive index. If None, uses config medium value.
        interaction : str, default='total'
            Energy loss mechanism: 'total', 'brems', 'pair', 'compton', etc.
        function : bool, default=False
            If True, returns callables instead of evaluated arrays.

        Returns
        -------
        TrackYieldResult
            Container with dcounts, angles, energy, and interaction attributes.

        Raises
        ------
        ValidationError
            If any input parameter is invalid, with helpful error message.

        Examples
        --------
        >>> fennel = Fennel()
        >>> result = fennel.track_yields_v2(100.0)
        >>> print(result)
        TrackYieldResult(energy=100.0 GeV, interaction='total', mode=array)
        >>> total_photons = integrate_trapezoid(result.dcounts, wavelengths)

        See Also
        --------
        track_yields : Original API method (still supported)
        quick_track : Simplified interface with minimal parameters
        """
        # Validate inputs with helpful error messages
        validate_energy(energy)
        wavelengths = validate_wavelengths(wavelengths)
        validate_interaction(interaction)
        n = validate_refractive_index(n)

        # Call original method
        dcounts, angles = self.track_yields(
            energy, wavelengths, angle_grid, n, interaction, function
        )

        # Return structured result
        return TrackYieldResult(
            dcounts=dcounts, angles=angles, energy=energy, interaction=interaction
        )

    def em_yields_v2(
        self,
        energy: float,
        particle: int,
        wavelengths: Optional[np.ndarray] = None,
        angle_grid: Optional[np.ndarray] = None,
        n: Optional[float] = None,
        z_grid: Optional[np.ndarray] = None,
        function: bool = False,
    ) -> EMYieldResult:
        """
        Calculate EM cascade light yields with enhanced API (v2.0).

        This is an improved version of em_yields() that returns a structured
        result container and includes comprehensive input validation with
        helpful error messages.

        Parameters
        ----------
        energy : float
            Cascade energy in GeV. Must be positive.
        particle : int
            PDG ID: 11 (e-), -11 (e+), 22 (γ)
        wavelengths : np.ndarray, optional
            Wavelength grid in nm. If None, uses config default.
        angle_grid : np.ndarray, optional
            Angular grid in radians. If None, uses config default.
        n : float, optional
            Refractive index. If None, uses config medium value.
        z_grid : np.ndarray, optional
            Longitudinal distance grid. If None, uses config default.
        function : bool, default=False
            If True, returns callables instead of evaluated arrays.

        Returns
        -------
        EMYieldResult
            Container with dcounts, dcounts_sample, longitudinal_profile,
            angles, energy, and particle attributes.

        Raises
        ------
        ValidationError
            If any input parameter is invalid, with helpful error message.

        Examples
        --------
        >>> fennel = Fennel()
        >>> result = fennel.em_yields_v2(1000.0, particle=11)
        >>> print(result)
        EMYieldResult(energy=1000.0 GeV, particle=electron, mode=array)
        >>> print(f"Particle: {result.particle_name}")
        Particle: electron

        See Also
        --------
        em_yields : Original API method (still supported)
        quick_cascade : Simplified interface with minimal parameters
        """
        # Validate inputs
        validate_energy(energy)
        validate_particle_pdg(particle, allowed_types=["em"])
        wavelengths = validate_wavelengths(wavelengths)
        n = validate_refractive_index(n)

        # Call original method
        dcounts, dcounts_sample, long_prof, angles = self.em_yields(
            energy, particle, wavelengths, angle_grid, n, z_grid, function
        )

        # Return structured result
        return EMYieldResult(
            dcounts=dcounts,
            dcounts_sample=dcounts_sample,
            longitudinal_profile=long_prof,
            angles=angles,
            energy=energy,
            particle=particle,
        )

    def hadron_yields_v2(
        self,
        energy: float,
        particle: int,
        wavelengths: Optional[np.ndarray] = None,
        angle_grid: Optional[np.ndarray] = None,
        n: Optional[float] = None,
        z_grid: Optional[np.ndarray] = None,
        function: bool = False,
    ) -> HadronYieldResult:
        """
        Calculate hadron cascade light yields with enhanced API (v2.0).

        This is an improved version of hadron_yields() that returns a structured
        result container and includes comprehensive input validation with
        helpful error messages.

        Parameters
        ----------
        energy : float
            Hadron energy in GeV. Must be positive.
        particle : int
            PDG ID: 211 (π+), -211 (π-), 130 (K_L), 2212 (p), 2112 (n)
        wavelengths : np.ndarray, optional
            Wavelength grid in nm. If None, uses config default.
        angle_grid : np.ndarray, optional
            Angular grid in radians. If None, uses config default.
        n : float, optional
            Refractive index. If None, uses config medium value.
        z_grid : np.ndarray, optional
            Longitudinal distance grid. If None, uses config default.
        function : bool, default=False
            If True, returns callables instead of evaluated arrays.

        Returns
        -------
        HadronYieldResult
            Container with dcounts, dcounts_sample, em_fraction,
            em_fraction_sample, longitudinal_profile, angles, energy,
            and particle attributes.

        Raises
        ------
        ValidationError
            If any input parameter is invalid, with helpful error message.

        Examples
        --------
        >>> fennel = Fennel()
        >>> result = fennel.hadron_yields_v2(1000.0, particle=211)
        >>> print(result)
        HadronYieldResult(energy=1000.0 GeV, particle=π+, em_frac=0.74, mode=array)
        >>> print(f"EM fraction: {result.em_fraction:.1%}")
        EM fraction: 74.0%

        See Also
        --------
        hadron_yields : Original API method (still supported)
        quick_cascade : Simplified interface with minimal parameters
        """
        # Validate inputs
        validate_energy(energy)
        validate_particle_pdg(particle, allowed_types=["hadron"])
        wavelengths = validate_wavelengths(wavelengths)
        n = validate_refractive_index(n)

        # Call original method
        dcounts, dcounts_sample, em_frac, em_frac_sample, long_prof, angles = (
            self.hadron_yields(
                energy, particle, wavelengths, angle_grid, n, z_grid, function
            )
        )

        # Return structured result
        return HadronYieldResult(
            dcounts=dcounts,
            dcounts_sample=dcounts_sample,
            em_fraction=em_frac,
            em_fraction_sample=em_frac_sample,
            longitudinal_profile=long_prof,
            angles=angles,
            energy=energy,
            particle=particle,
        )

    # =========================================================================
    # Convenience Methods for Common Use Cases
    # =========================================================================

    def quick_track(
        self, energy: float, interaction: str = "total"
    ) -> TrackYieldResult:
        """
        Quick track calculation with minimal parameters.

        Uses sensible defaults for wavelengths, angles, and refractive index
        from configuration. Perfect for quick calculations or when you don't
        need to customize the grids.

        Parameters
        ----------
        energy : float
            Particle energy in GeV
        interaction : str, default='total'
            Energy loss mechanism

        Returns
        -------
        TrackYieldResult
            Result container with default grids

        Examples
        --------
        >>> fennel = Fennel()
        >>> result = fennel.quick_track(100.0)
        >>> result = fennel.quick_track(100.0, interaction='brems')
        """
        validate_energy(energy)
        validate_interaction(interaction)
        return self.track_yields_v2(energy, interaction=interaction)

    def quick_cascade(
        self, energy: float, particle: int
    ) -> Union[EMYieldResult, HadronYieldResult]:
        """
        Quick cascade calculation with minimal parameters.

        Automatically detects whether the particle is EM or hadronic and calls
        the appropriate method. Uses sensible defaults from configuration.

        Parameters
        ----------
        energy : float
            Cascade energy in GeV
        particle : int
            PDG ID of the particle

        Returns
        -------
        EMYieldResult or HadronYieldResult
            Appropriate result container based on particle type

        Examples
        --------
        >>> fennel = Fennel()
        >>> electron_result = fennel.quick_cascade(1000.0, particle=11)
        >>> pion_result = fennel.quick_cascade(1000.0, particle=211)
        """
        validate_energy(energy)
        validate_particle_pdg(particle)

        # Determine particle type
        em_particles = config.get("simulation", {}).get("em particles", [])
        hadron_particles = config.get("simulation", {}).get("hadron particles", [])

        if particle in em_particles:
            return self.em_yields_v2(energy, particle)
        elif particle in hadron_particles:
            return self.hadron_yields_v2(energy, particle)
        else:
            raise ValueError(
                f"Particle {particle} not recognized as EM or hadron cascade. "
                f"{suggest_particle_type(particle)}"
            )

    def calculate(
        self,
        energy: float,
        particle: Optional[int] = None,
        particle_type: Optional[str] = None,
        interaction: str = "total",
    ) -> Union[TrackYieldResult, EMYieldResult, HadronYieldResult]:
        """
        Universal calculation method that auto-detects particle type.

        This is the most flexible method - you can specify either a PDG ID
        or a particle type name, and it will call the appropriate calculation.

        Parameters
        ----------
        energy : float
            Particle/cascade energy in GeV
        particle : int, optional
            PDG ID of the particle. If provided, type is auto-detected.
        particle_type : str, optional
            Particle type: 'muon'/'track', 'electron'/'em', 'pion'/'hadron'
            Only used if particle PDG ID is not provided.
        interaction : str, default='total'
            For tracks: energy loss mechanism

        Returns
        -------
        TrackYieldResult, EMYieldResult, or HadronYieldResult
            Appropriate result container based on particle type

        Raises
        ------
        ValueError
            If neither particle nor particle_type is specified, or if both
            are specified and conflict.

        Examples
        --------
        >>> fennel = Fennel()
        >>> # Auto-detect from PDG ID
        >>> result = fennel.calculate(100.0, particle=11)  # electron
        >>> result = fennel.calculate(100.0, particle=211)  # pion
        >>>
        >>> # Specify by name (uses default PDG for that type)
        >>> result = fennel.calculate(100.0, particle_type='muon')
        >>> result = fennel.calculate(100.0, particle_type='electron')
        """
        validate_energy(energy)

        if particle is None and particle_type is None:
            raise ValueError(
                "Must specify either 'particle' (PDG ID) or 'particle_type'. "
                "Examples:\n"
                "  fennel.calculate(100.0, particle=11)  # electron\n"
                "  fennel.calculate(100.0, particle_type='muon')"
            )

        # If particle type name is given, convert to PDG ID
        if particle is None and particle_type is not None:
            type_map = {
                "muon": 13,
                "track": 13,
                "electron": 11,
                "em": 11,
                "e-": 11,
                "positron": -11,
                "e+": -11,
                "photon": 22,
                "gamma": 22,
                "pion": 211,
                "pi+": 211,
                "hadron": 211,
                "proton": 2212,
                "p": 2212,
                "neutron": 2112,
                "n": 2112,
            }
            particle = type_map.get(particle_type.lower())
            if particle is None:
                raise ValueError(
                    f"Unknown particle_type: '{particle_type}'. "
                    f"Valid options: {list(type_map.keys())}"
                )

        # Now route to appropriate method
        validate_particle_pdg(particle)

        track_particles = config.get("simulation", {}).get("track particles", [])
        em_particles = config.get("simulation", {}).get("em particles", [])
        hadron_particles = config.get("simulation", {}).get("hadron particles", [])

        if particle in track_particles:
            return self.track_yields_v2(energy, interaction=interaction)
        elif particle in em_particles:
            return self.em_yields_v2(energy, particle)
        elif particle in hadron_particles:
            return self.hadron_yields_v2(energy, particle)
        else:
            raise ValueError(f"Particle {particle} not recognized")

    def definitions(self):
        """Write the definitions file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._dg._write()

    def pars2csv(self):
        """Write the parameters to a csv file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._dg._pars2csv()

    def hidden_function(self):
        """Yaha! You found me!"""
        print("                  @*****&@         @@.                    ")
        print("           @@@((@ @*******@     @%((((@@((@               ")
        print("         @(((((((@(@*******@@@((@(((((((((@*              ")
        print("        @((((((@(((@*******@(((@%(((((&%(((@              ")
        print("         #@((@((&@((&*******@((@(((((@#(((@               ")
        print("        @@*****@((#@@********@@(((((@(((@/**@@            ")
        print("        @********@((@@*************@@@#@******&%          ")
        print("@@  @ @&(@(***,*******,*******,*******,*******,@      .@@*")
        print(" @   @@((((@*************.*********...*******@(@      ./  ")
        print(" @     %(((#@**********...*,.. ........**,**(@(#*     @   ")
        print("  @      @@,.*.***,@@@.......*,...@@@..*,.....@@    &@    ")
        print("   @(    @....,,...@@......... ,*.@%...*......,%  @@      ")
        print("       @@@.....,***...............,****.......*@,         ")
        print("        .........*............ ........****..*.@          ")
        print("         #*........*...&       @...............@          ")
        print("         @ *.. ... ..,,..@@@@. ... ... ... ...&           ")
        print("          @.,*...........*,...................@           ")
        print("            @.**.............*,..............@            ")
        print("             .@.**...............**.........@             ")
        print("           @@@%   &@@,........ .......**#@    @@          ")
        print("        @,                 (@@@@@@@@(             @       ")
        print("                                                %@*       ")
