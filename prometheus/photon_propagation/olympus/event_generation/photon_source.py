from enum import Enum, auto


class PhotonSourceType(Enum):
    """Enum for photon source types."""

    STANDARD_CHERENKOV = auto()
    ISOTROPIC = auto()


class PhotonSource(object):
    """Representation of a single point-like photon source.

    Attributes
    ----------
    position : np.ndarray
        Position vector (shape (3,)).
    n_photons : int
        Number of photons emitted by the source.
    time : float
        Emission time.
    direction : np.ndarray
        Direction vector (shape (3,)).
    type : PhotonSourceType, optional
        Source type (default is ``PhotonSourceType.STANDARD_CHERENKOV``).
    """

    def __init__(
        self,
        position,
        n_photons,
        time,
        direction,
        type=PhotonSourceType.STANDARD_CHERENKOV,
    ):
        """Initialize the PhotonSource instance.

        Parameters are stored as instance attributes.
        """
        self.position = position
        self.n_photons = n_photons
        self.time = time
        self.direction = direction
        self.type = type
