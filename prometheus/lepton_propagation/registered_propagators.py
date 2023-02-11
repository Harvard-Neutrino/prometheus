from ..utils import ExtendedEnum

class RegisteredPropagators(ExtendedEnum):
    """Enum for tracking which lepton propagators we know how to handle"""
    OLDPROPOSAL=1
    NEWPROPOSAL=2
