from enum import Enum

class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.name, cls))
