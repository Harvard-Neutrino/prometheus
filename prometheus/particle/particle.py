# -*- coding: utf-8 -*-
# particle.py
# Copyright (C) 2022 Jeffrey Lazar
# Storage class for particles

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from ..utils.translators import PDG_to_pstring

@dataclass
class Particle:
    """Base dataclass for particle event structure

    fields
    ______
    pdg_code: PDG mc code
    e: energy in GeV
    position: particle position in meters
    direction: unit vector pointing along particle momentum
    serialization_idx: Index helper for serialization. This
        will be overwritten at serialization time
    """
    pdg_code: int
    e: float
    position: np.ndarray
    direction: np.ndarray
    serialization_idx: int


    def __str__(self):
        return PDG_to_pstring[self.pdg_code]

    def __int__(self):
        return int(self.pdg_code)

    @property
    def theta(self):
        return np.arccos(self.direction[2])

    @property
    def phi(self):
        return np.arctan2(self.direction[1], self.direction[0])

@dataclass
class PropagatableParticle(Particle):
    """Particle event structure with added feature to support propagation
    
    fields
    ______
    parent: Particle which created this particle
    children: Particles that this one spawned
    losses: energy losses created by this particle
    hits: OM hits originating from this particle
    """
    parent: Particle
    children: List[Particle] = field(default_factory=list)
    losses: List = field(default_factory=list)
    hits: List = field(default_factory=list)
