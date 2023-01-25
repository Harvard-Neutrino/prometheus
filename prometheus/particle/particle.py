# -*- coding: utf-8 -*-
# particle.py
# Copyright (C) 2022 Jeffrey Lazar
# Storage class for particles

import numpy as np
import proposal as pp
if int(pp.__version__.split(".")[0]) >= 7:
    from proposal import Cartesian3D as pp_vector
else:
    from proposal import Vector3D as pp_vector

from ..utils.units import m_to_cm, GeV_to_MeV
from ..utils.translators import PDG_to_pstring

class Particle(object):
    """Base class for particle event structure"""
    def __init__(
        self,
        pdg_code: int,
        e: float,
        position: np.ndarray,
        direction: np.ndarray,
    ):
        """Construct Particle

        params
        ______
        pdg_code: PDG mc code
        e: energy in GeV
        position:


        returns
        _______

        """
        self._e = e
        self._position = np.array(position)
        self._pp_position = pp_vector(*(m_to_cm*position))
        self._direction = np.array(direction)
        self._pp_direction = pp_vector(*direction)
        self._theta = np.arccos(self._direction[2])
        self._phi = np.arctan2(self.direction[1], self.direction[0])
        self._pdg_code = pdg_code
        self._str = PDG_to_pstring[pdg_code]
        self.serialization_idx = None


    def __str__(self):
        return self._str

    def __int__(self):
        return int(self._pdg_code)

    def __repr__(self):
        s = f"pdg_code : {self._pdg_code}\n"
        s += f"name : {self._str}\n"
        s += f"e : {self._e}\n"
        s += f"position : {self._position}\n"
        s += f"direction : {self._direction}\n"
        return s

    @property
    def e(self):
        return self._e

    @property
    def pdg_code(self):
        return self._pdg_code

    @property
    def position(self):
        return self._position

    @property
    def direction(self):
        return self._direction

    @property
    def theta(self):
        return self._theta

    @property
    def phi(self):
        return self._phi

    @property
    def pp_direction(self):
        return self._pp_direction

    @property
    def pp_position(self):
        return self._pp_position



class PropagatableParticle(Particle):
    """Particle event structure with added feature to support propagation"""
    def __init__(
        self,
        pdg_code: int,
        e: float,
        position: np.ndarray,
        direction: np.ndarray,
        parent=None
    ):
        super().__init__(pdg_code, e, position, direction)
        self._parent = parent
        self._children = []
        self._losses = []
        self._hits = []

    @property
    def losses(self):
        return self._losses

    @property
    def hits(self):
        return self._hits

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    def add_loss(self, loss):
        self._losses.append(loss)

    def add_child(self, child):
        self._children.append(child)
