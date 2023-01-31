from ..utils import ExtendedEnum

class RegisteredInjectors(ExtendedEnum):
    """Enum specifying allowed injectors"""
    LEPTONINJECTOR=1
    PROMETHEUS=2
    GENIE=3
