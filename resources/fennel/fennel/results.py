# -*- coding: utf-8 -*-
"""
Result container classes for light yield calculations.

This module provides structured result containers that make it easier to work
with the outputs from fennel calculations. These are returned when using the
newer API methods.

Classes
-------
TrackYieldResult : Container for track light yields
EMYieldResult : Container for electromagnetic cascade yields
HadronYieldResult : Container for hadronic cascade yields
"""

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np


@dataclass
class TrackYieldResult:
    """
    Result container for track light yield calculations.

    Attributes
    ----------
    dcounts : np.ndarray or Callable
        Differential photon counts per unit wavelength and unit path length.
        Shape: (n_wavelengths,) if array, or callable(energy, wavelengths)
    angles : np.ndarray or Callable
        Angular distribution of emitted Cherenkov light.
        Shape: (n_angles,) if array, or callable(angle_grid, n, energy)
    energy : float
        Particle energy in GeV
    interaction : str
        Type of interaction ('total', 'brems', 'pair', 'compton', etc.)

    Examples
    --------
    >>> result = fennel.track_yields_v2(100.0)
    >>> total_photons = integrate_trapezoid(result.dcounts, wavelengths)
    >>> print(f"Energy: {result.energy} GeV")
    >>> print(f"Interaction: {result.interaction}")
    """

    dcounts: Union[np.ndarray, Callable]
    angles: Union[np.ndarray, Callable]
    energy: float
    interaction: str

    @property
    def is_function(self) -> bool:
        """Check if results are callables (function=True) or arrays."""
        return callable(self.dcounts)

    def __repr__(self) -> str:
        mode = "functional" if self.is_function else "array"
        return (
            f"TrackYieldResult(energy={self.energy} GeV, "
            f"interaction='{self.interaction}', mode={mode})"
        )


@dataclass
class EMYieldResult:
    """
    Result container for electromagnetic cascade light yield calculations.

    Attributes
    ----------
    dcounts : np.ndarray or Callable
        Mean differential photon counts.
        Shape: (n_wavelengths,) if array
    dcounts_sample : np.ndarray or Callable
        Sampled differential photon counts for stochastic modeling.
        Shape: (n_wavelengths,) if array
    longitudinal_profile : np.ndarray or Callable
        Longitudinal shower development profile along propagation direction.
        Shape: (n_z_positions,) if array
    angles : np.ndarray or Callable
        Angular distribution of emitted Cherenkov light.
        Shape: (n_angles,) if array
    energy : float
        Cascade energy in GeV
    particle : int
        PDG ID of the particle (11: e-, -11: e+, 22: γ)

    Examples
    --------
    >>> result = fennel.em_yields_v2(1000.0, particle=11)
    >>> print(f"Electron cascade at {result.energy} GeV")
    >>> shower_length = np.argmax(result.longitudinal_profile)
    """

    dcounts: Union[np.ndarray, Callable]
    dcounts_sample: Union[np.ndarray, Callable]
    longitudinal_profile: Union[np.ndarray, Callable]
    angles: Union[np.ndarray, Callable]
    energy: float
    particle: int

    @property
    def is_function(self) -> bool:
        """Check if results are callables (function=True) or arrays."""
        return callable(self.dcounts)

    @property
    def particle_name(self) -> str:
        """Get human-readable particle name."""
        names = {11: "electron", -11: "positron", 22: "photon"}
        return names.get(self.particle, f"PDG_{self.particle}")

    def __repr__(self) -> str:
        mode = "functional" if self.is_function else "array"
        return (
            f"EMYieldResult(energy={self.energy} GeV, "
            f"particle={self.particle_name}, mode={mode})"
        )


@dataclass
class HadronYieldResult:
    """
    Result container for hadronic cascade light yield calculations.

    Attributes
    ----------
    dcounts : np.ndarray or Callable
        Mean differential photon counts.
        Shape: (n_wavelengths,) if array
    dcounts_sample : np.ndarray or Callable
        Sampled differential photon counts for stochastic modeling.
        Shape: (n_wavelengths,) if array
    em_fraction : float or Callable
        Electromagnetic fraction of the cascade (0.0 to 1.0).
        Typically increases with energy.
    em_fraction_sample : float or Callable
        Sampled electromagnetic fraction for stochastic modeling.
    longitudinal_profile : np.ndarray or Callable
        Longitudinal shower development profile.
        Shape: (n_z_positions,) if array
    angles : np.ndarray or Callable
        Angular distribution of emitted Cherenkov light.
        Shape: (n_angles,) if array
    energy : float
        Hadron energy in GeV
    particle : int
        PDG ID of the hadron (211: π+, -211: π-, 130: K_L, 2212: p, 2112: n)

    Examples
    --------
    >>> result = fennel.hadron_yields_v2(1000.0, particle=211)
    >>> print(f"Pion cascade: EM fraction = {result.em_fraction:.2%}")
    >>> print(f"Energy: {result.energy} GeV")
    """

    dcounts: Union[np.ndarray, Callable]
    dcounts_sample: Union[np.ndarray, Callable]
    em_fraction: Union[float, Callable]
    em_fraction_sample: Union[float, Callable]
    longitudinal_profile: Union[np.ndarray, Callable]
    angles: Union[np.ndarray, Callable]
    energy: float
    particle: int

    @property
    def is_function(self) -> bool:
        """Check if results are callables (function=True) or arrays."""
        return callable(self.dcounts)

    @property
    def particle_name(self) -> str:
        """Get human-readable particle name."""
        names = {
            211: "π+",
            -211: "π-",
            130: "K_L",
            321: "K+",
            -321: "K-",
            2212: "proton",
            -2212: "antiproton",
            2112: "neutron",
            -2112: "antineutron",
        }
        return names.get(self.particle, f"PDG_{self.particle}")

    def __repr__(self) -> str:
        mode = "functional" if self.is_function else "array"
        em_str = "callable" if callable(self.em_fraction) else f"{self.em_fraction:.3f}"
        return (
            f"HadronYieldResult(energy={self.energy} GeV, "
            f"particle={self.particle_name}, em_frac={em_str}, "
            f"mode={mode})"
        )
