# -*- coding: utf-8 -*-
"""
Particle class definitions for fennel.

This module defines the Particle class which encapsulates particle properties
from the PDG (Particle Data Group) numbering scheme.
"""

import logging

from .config import config

_log = logging.getLogger(__name__)


class Particle:
    """
    Particle object representing a physics particle.

    This class stores particle properties such as PDG ID, name, mass,
    and standard track length for use in light yield calculations.

    Parameters
    ----------
    pdg_id : int
        The PDG (Particle Data Group) identification number of the particle.
        Supported particles include electrons (11), positrons (-11), muons (13, -13),
        photons (22), pions (211, -211), kaons (130), protons (2212, -2212), and neutrons (2112).

    Attributes
    ----------
    _pdg_id : int
        The PDG identification number
    _name : str
        Particle name following PDG Monte Carlo naming convention
    _energies : np.ndarray
        Energy grid for calculations
    _mass : float, optional
        Particle rest mass in GeV (for muons)
    _std_track : float, optional
        Standard track length in cm (for muons)

    Examples
    --------
    >>> from fennel.particle import Particle
    >>> muon = Particle(13)
    >>> print(muon._name)
    'mu-'

    Notes
    -----
    Currently, only muons have their mass and standard track length implemented.
    """

    def __init__(self, pdg_id: int) -> None:
        """
        Initialize a Particle object.

        Parameters
        ----------
        pdg_id : int
            The PDG number of the particle

        Raises
        ------
        KeyError
            If the PDG ID is not recognized in the configuration
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.info("Constructing a particle")
        self._pdg_id = pdg_id
        # Naming conventions PDG Monte Carlo scheme
        self._name = config["pdg id"][pdg_id]
        _log.debug("The final name is " + self._name)
        self._energies = config["advanced"]["energy grid"]
        # TODO: Add masses for all particles
        # Masses of the muons
        if self._name[:2] == "mu":
            self._mass = config[self._name]["mass"]
            self._std_track = config[self._name]["standard track length"]
