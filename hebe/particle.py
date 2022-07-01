import numpy as np
from proposal import Vector3D
from hebe.utils.units import m_to_cm, cm_to_m, GeV_to_MeV, MeV_to_GeV
from hebe.utils.translators import PDG_to_pstring

class Particle(object):

    def __init__(self, pdg_code, e, position, direction, parent=None):
        if isinstance(position, Vector3D):
            if not isinstance(direction, Vector3D):
                raise ValueError()
            self._e = e * MeV_to_GeV
            self._position = cm_to_m * np.array([position.x, position.y, position.z])
            self._pp_position = position
            self._direction = np.array([direction.x, direction.y, direction.z])
            self._pp_direction = direction
        else:
            self._e = e
            self._position = np.array(position)
            self._pp_position = Vector3D(*(m_to_cm*position))
            self._direction = np.array(direction)
            self._pp_direction = Vector3D(*direction)
        self._pdg_code = pdg_code
        self._parent = parent
        self._children = []
        self._losses = []
        self._str = PDG_to_pstring[pdg_code]

    def __str__(self):
        return self._str

    def __int__(self):
        return int(self._pdg_code)

    def __repr__(self):
        return f"""pdg_code : {self._pdg_code}
        name : {self._str}
        e : {self._e}
        position : {self._position}
        direction : {self._direction}
        children : {self._children}"""

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
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @property
    def pp_direction(self):
        return self._pp_direction

    @property
    def pp_position(self):
        return self._pp_position

    @property
    def losses(self):
        return self._losses

    def add_loss(self, loss):
        self._losses.append(loss)

    def add_child(self, child):
        self._children.append(child)
