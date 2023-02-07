import numpy as np
from typing import Tuple

class Module:
    """Detector optical module"""
    def __init__(
        self,
        pos: np.ndarray, 
        key: Tuple[int, int],
        noise_rate: int=1e3,
        efficiency=0.2,
        serial_no=None
    ):
        """Initialize a module.

        params
        ______
        pos: position of the optical module in meters
        key: tuple to look up module by. (string index, om index) is the convention
        [noise_rate]: noise of the module in GHz
        [efficiency]: quantum efficiency of module
        [serial number]: Serial number for the OM. I don't think you ever need to
            touch this, but I don't want to tickle a sleeping dragon
        """
        self.pos = pos
        self.noise_rate = noise_rate
        self.efficiency = efficiency
        self.key = key
        self.serial_no = serial_no

    def __repr__(self):
        """Return string representation."""
        return repr(
            f"Module {self.key}, {self.pos} [m], {self.noise_rate*1e-9} [Hz], {self.efficiency}"
        )


