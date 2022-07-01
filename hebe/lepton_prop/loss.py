import numpy as np
from proposal import Vector3D

from ..utils import int_type_to_str

class Loss(object):

    def __init__(self, int_type, e, position):
        self._int_type = int_type
        self._e = e
        if isinstance(position, Vector3D):
            self._position = np.array(
                [position.x * cm_to_m, position.y * cm_to_m, position.z * cm_to_m
            ])
        else:
            self._position = np.array(position)
        self._str = int_type_to_str[int_type]

    def __str__(self):
        return self._str

    def __rep__(self):
        return self._str

    @property
    def int_type(self):
        return self._int_type

    @property
    def e(self):
        return self._e

    @property
    def position(self):
        return self._position
