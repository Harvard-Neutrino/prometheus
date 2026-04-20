from ..utils import ExtendedEnum

class RegisteredPropagators(ExtendedEnum):
    """Enum to keep track of which photon propagators we know about.
    
    Notes
    -----
    Members: ``OLYMPUS``, ``PPC``, ``PPCCUDA``.
    """
    OLYMPUS=1
    PPC=2
    PPCCUDA=3
