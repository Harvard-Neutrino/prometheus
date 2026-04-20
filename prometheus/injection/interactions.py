from enum import Enum

class Interactions(Enum):
    """Enum of known interaction types.
    
    Notes
    -----
    Members: ``GLASHOW_RESONANCE``, ``CHARGED_CURRENT``, ``NEUTRAL_CURRENT``, ``DIMUON``.
    """
    GLASHOW_RESONANCE = 0
    CHARGED_CURRENT = 1
    NEUTRAL_CURRENT = 2
    DIMUON = 3
