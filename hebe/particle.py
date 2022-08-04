import numpy as np
import proposal as pp
if int(pp.__version__.split('.')[0])>=7:
    from proposal import Cartesian3D as pp_vector
else:
    from proposal import Vector3D as pp_vector
from hebe.utils.units import m_to_cm, cm_to_m, GeV_to_MeV, MeV_to_GeV
from hebe.utils.translators import PDG_to_pstring

class Particle(object):

    def __init__(self,
        pdg_code, e, position,
        direction, event_id,
        theta=0., phi=0.,
        parent=None):
        if isinstance(position, pp_vector):
            if not isinstance(direction, pp_vector):
                raise ValueError()
            self._e = e * MeV_to_GeV
            self._position = cm_to_m * np.array([position.x, position.y, position.z])
            self._pp_position = position
            self._direction = np.array([direction.x, direction.y, direction.z])
            self._pp_direction = direction
            self._theta = theta
            self._phi = phi
        else:
            self._e = e
            self._position = np.array(position)
            self._pp_position = pp_vector(*(m_to_cm*position))
            self._direction = np.array(direction)
            self._pp_direction = pp_vector(*direction)
            self._theta = theta
            self._phi = phi
        self._pdg_code = pdg_code
        self._parent = parent
        self._children = []
        self._losses = []
        self._event_id = event_id
        self._hits = []
        self._str = PDG_to_pstring[pdg_code]

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
        s += f"children : {self._children}"
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

    @property
    def hits(self):
        return self._hits

    @property
    def event_id(self):
        return self._event_id

    def add_loss(self, loss):
        self._losses.append(loss)

    def add_child(self, child):
        self._children.append(child)
