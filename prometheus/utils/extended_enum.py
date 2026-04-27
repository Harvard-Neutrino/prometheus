from enum import Enum


class ExtendedEnum(Enum):
    """Enum subclass with a convenience ``list`` class method."""

    @classmethod
    def list(cls):
        """Return a list of all member names.

        Returns
        -------
        names : list of str
            Names of all enum members.
        """
        return list(map(lambda c: c.name, cls))
