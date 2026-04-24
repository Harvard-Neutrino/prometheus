import random
import string


def random_serial() -> str:
    """Create a serial number in the hexadecimal form expected by ppc.

    Returns
    -------
    serial : str
        Hexadecimal OM serial number.
    """
    serial = "0x" + "".join(random.choices("0123456789abcdef", k=12))
    return serial


def random_mac() -> str:
    """Creates a random MAC ID in the form expected by ppc.

    Returns
    -------
    mac : str
        MAC ID.
    """
    mac = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return mac
