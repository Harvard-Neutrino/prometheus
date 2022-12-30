# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,

import numpy as np
import proposal as pp

from .lepton_propagator import LeptonPropagator
from ..particle import Particle
from ..detector import Detector

class NewProposalLeptonPropagator(LeptonPropagator):
    """Class for propagating charged leptons with PROPOSAL versions >= 7"""
    def __init__(self, config):
        super().__init__(config)

    def _make_propagator(self, particle: Particle) -> pp.Propagator:
        """Make a PROPOSAL propagator

        params
        ______
        particle: Prometheus Particle that you want a PROPOSAL propagator for

        returns
        _______
        propagator: PROPOSAL propagator
        """ 
        propagator = None
        return propagator

    def _make_particle_def(self, particle: Particle):
        """Make a PROPOSAL ParticleDef

        params
        ______
        particle: Prometheus Particle that you want a PROPOSAL ParticleDef for

        returns
        _______
        pdef: PROPOSAL ParticleDef
        """ 
        pdef = None
        return pdef

    def energy_losses(
        self, 
        particle: Particle,
        detector: Detector
    ) -> Particle:
        """Propagate a particle and track the losses

        params
        ______
        particle: Prometheus Particle that should be propagated
        detector: Detector that this is being propagated within
            This is a temporary fix and will hopefully we solved soon :-)

        returns
        _______
        propped_particle: Prometheus Particle after propagation
        """
        propped_particle = None
        return propped_particle
