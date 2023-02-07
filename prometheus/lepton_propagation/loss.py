import numpy as np
from dataclasses import dataclass

from ..utils import int_type_to_str

@dataclass(frozen=True)
class Loss:
    """Dataclass for handling PROPOSAL energy losses
    params
    ______
    int_type: interaction type
    e: energy lost
    position: position of the loss in meters
    """
    int_type: int
    e: float
    position: np.ndarray

    def __str__(self):
        return int_type_to_str[self.int_type]
