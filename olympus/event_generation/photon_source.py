from enum import Enum, auto


class PhotonSourceType(Enum):
    """Enum for photon source types."""

    STANDARD_CHERENKOV = auto()
    ISOTROPIC = auto()


class PhotonSource(object):
    """
    This class represents a single (pointlike) source of photons.
    """

    def __init__(
        self,
        position,
        n_photons,
        time,
        direction,
        type=PhotonSourceType.STANDARD_CHERENKOV,
    ):
        """Initialize PhotonSource_."""
        self.position = position
        self.n_photons = n_photons
        self.time = time
        self.direction = direction
        self.type = type
