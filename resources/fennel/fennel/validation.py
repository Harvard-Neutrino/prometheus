# -*- coding: utf-8 -*-
"""
Input validation utilities for fennel calculations.

This module provides validation functions that check user inputs and provide
clear, helpful error messages with suggestions for fixes.
"""

from typing import List, Optional

import numpy as np

from .config import config


class ValidationError(ValueError):
    """Custom exception for validation errors with helpful messages."""

    pass


def validate_energy(energy: float, min_energy: float = 0.0) -> None:
    """
    Validate particle/cascade energy.

    Parameters
    ----------
    energy : float
        Energy in GeV to validate
    min_energy : float, optional
        Minimum allowed energy in GeV (default: 0.0)

    Raises
    ------
    ValidationError
        If energy is invalid
    """
    if not isinstance(energy, (int, float, np.number)):
        raise ValidationError(
            f"Energy must be a number, got {type(energy).__name__}. "
            f"Example: energy=100.0"
        )

    if energy <= min_energy:
        raise ValidationError(
            f"Energy must be positive, got {energy} GeV. " f"Example: energy=100.0"
        )

    if energy > 1e12:
        raise ValidationError(
            f"Energy {energy:.2e} GeV seems unrealistically high. "
            f"Did you mean {energy/1000:.2e} GeV? "
            f"Note: Energy should be in GeV."
        )


def validate_particle_pdg(
    particle: int, allowed_types: Optional[List[str]] = None
) -> None:
    """
    Validate PDG particle ID.

    Parameters
    ----------
    particle : int
        PDG ID to validate
    allowed_types : list of str, optional
        List of allowed particle types: 'track', 'em', 'hadron'
        If None, allows all types configured in config

    Raises
    ------
    ValidationError
        If particle ID is invalid or not allowed
    """
    if not isinstance(particle, (int, np.integer)):
        raise ValidationError(
            f"Particle must be an integer PDG ID, got {type(particle).__name__}. "
            f"Example: particle=11 (electron)"
        )

    # Get all known particles from config
    track_particles = config.get("simulation", {}).get("track particles", [])
    em_particles = config.get("simulation", {}).get("em particles", [])
    hadron_particles = config.get("simulation", {}).get("hadron particles", [])

    all_particles = set(track_particles + em_particles + hadron_particles)

    if particle not in all_particles:
        # Provide helpful suggestions
        common_particles = {
            13: "muon (μ-)",
            -13: "antimuon (μ+)",
            11: "electron (e-)",
            -11: "positron (e+)",
            22: "photon (γ)",
            211: "positive pion (π+)",
            -211: "negative pion (π-)",
            130: "K_L (neutral kaon)",
            2212: "proton (p)",
            2112: "neutron (n)",
        }

        suggestion = ""
        if abs(particle) in common_particles:
            suggestion = (
                f"\nDid you mean {common_particles[abs(particle)]} ({abs(particle)})?"
            )

        raise ValidationError(
            f"Unknown particle PDG ID: {particle}. "
            f"Supported particles: {sorted(all_particles)}.{suggestion}\n"
            f"Common examples:\n"
            f"  Tracks: 13 (muon)\n"
            f"  EM cascades: 11 (electron), -11 (positron), 22 (photon)\n"
            f"  Hadron cascades: 211 (π+), 2212 (proton), 2112 (neutron)"
        )

    # Check if particle type is allowed
    if allowed_types is not None:
        particle_type = None
        if particle in track_particles:
            particle_type = "track"
        elif particle in em_particles:
            particle_type = "em"
        elif particle in hadron_particles:
            particle_type = "hadron"

        if particle_type not in allowed_types:
            raise ValidationError(
                f"Particle {particle} is a {particle_type} particle, "
                f"but only {allowed_types} particles are allowed for this method. "
                f"Use the appropriate method: "
                f"track_yields() for tracks, "
                f"em_yields() for EM cascades, "
                f"hadron_yields() for hadron cascades."
            )


def validate_wavelengths(wavelengths: Optional[np.ndarray]) -> np.ndarray:
    """
    Validate wavelength array.

    Parameters
    ----------
    wavelengths : array-like or None
        Wavelengths in nm

    Returns
    -------
    np.ndarray
        Validated wavelength array

    Raises
    ------
    ValidationError
        If wavelengths are invalid
    """
    if wavelengths is None:
        return config["advanced"]["wavelengths"]

    wavelengths = np.asarray(wavelengths)

    if wavelengths.ndim != 1:
        raise ValidationError(
            f"Wavelengths must be 1D array, got shape {wavelengths.shape}. "
            f"Example: wavelengths=np.linspace(300, 600, 100)"
        )

    if len(wavelengths) == 0:
        raise ValidationError(
            "Wavelengths array is empty. "
            "Example: wavelengths=np.linspace(300, 600, 100)"
        )

    if np.any(wavelengths <= 0):
        raise ValidationError(
            f"Wavelengths must be positive, got min={wavelengths.min():.2f} nm. "
            f"Note: Wavelengths should be in nanometers (nm)."
        )

    if np.any(wavelengths > 2000):
        raise ValidationError(
            f"Wavelengths seem too large (max={wavelengths.max():.2f} nm). "
            f"Typical range: 300-600 nm. "
            f"Note: Wavelengths should be in nanometers (nm), not meters."
        )

    return wavelengths


def validate_interaction(interaction: str) -> None:
    """
    Validate interaction type for track calculations.

    Parameters
    ----------
    interaction : str
        Interaction type to validate

    Raises
    ------
    ValidationError
        If interaction type is invalid
    """
    valid_interactions = [
        "total",
        "brems",
        "pair",
        "photo",
        "compton",
        "delta",
        "mubrems",
        "muhad",
        "mupair",
    ]

    if not isinstance(interaction, str):
        raise ValidationError(
            f"Interaction must be a string, got {type(interaction).__name__}. "
            f"Example: interaction='total'"
        )

    if interaction not in valid_interactions:
        raise ValidationError(
            f"Unknown interaction type: '{interaction}'. "
            f"Valid options: {valid_interactions}. "
            f"Most common: 'total' (all interactions), 'brems' (bremsstrahlung), "
            f"'pair' (pair production)"
        )


def validate_refractive_index(n: Optional[float]) -> float:
    """
    Validate refractive index.

    Parameters
    ----------
    n : float or None
        Refractive index

    Returns
    -------
    float
        Validated refractive index

    Raises
    ------
    ValidationError
        If refractive index is invalid
    """
    if n is None:
        medium = config["scenario"]["medium"]
        return config["mediums"][medium]["refractive index"]

    if not isinstance(n, (int, float, np.number)):
        raise ValidationError(
            f"Refractive index must be a number, got {type(n).__name__}. "
            f"Example: n=1.33 (water) or n=1.32 (ice)"
        )

    if n < 1.0:
        raise ValidationError(
            f"Refractive index must be >= 1.0, got {n}. "
            f"Examples: n=1.33 (water), n=1.32 (ice), n=1.0 (vacuum)"
        )

    if n > 3.0:
        raise ValidationError(
            f"Refractive index {n} seems too high for typical transparent media. "
            f"Examples: n=1.33 (water), n=1.32 (ice). "
            f"Did you enter the wrong value?"
        )

    return n


def suggest_particle_type(particle: int) -> str:
    """
    Suggest the appropriate method for a given particle type.

    Parameters
    ----------
    particle : int
        PDG ID

    Returns
    -------
    str
        Suggestion message
    """
    track_particles = config.get("simulation", {}).get("track particles", [])
    em_particles = config.get("simulation", {}).get("em particles", [])
    hadron_particles = config.get("simulation", {}).get("hadron particles", [])

    if particle in track_particles:
        return f"Use track_yields() for particle {particle} (track particle)"
    elif particle in em_particles:
        return f"Use em_yields() for particle {particle} (EM cascade)"
    elif particle in hadron_particles:
        return f"Use hadron_yields() for particle {particle} (hadron cascade)"
    else:
        return f"Particle {particle} is not recognized"
