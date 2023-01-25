class Module(object):
    """
    Detection module.
    Attributes:
        pos: np.ndarray
            Module position (x, y, z)
        noise_rate: float
            Noise rate in 1/ns
        efficiency: float
            Module efficiency (0, 1]
        self.key: collection
            Module identifier
    """

    def __init__(self, pos, key, noise_rate=1, efficiency=0.2, serial_no=None):
        """Initialize a module."""
        self.pos = pos
        self.noise_rate = noise_rate
        self.efficiency = efficiency
        self.key = key
        self.serial_no = serial_no

    def __repr__(self):
        """Return string representation."""
        return repr(
            f"Module {self.key}, {self.pos} [m], {self.noise_rate} [Hz], {self.efficiency}"
        )


