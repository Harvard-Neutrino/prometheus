import random
import string

def random_serial() -> str:
    """Makes a serial number in the hexdecimal form expected by PPC

    returns
    _______
    serial: Hexadecimal OM serial number
    """
    serial = "0x"+"".join(random.choices('0123456789abcdef', k=12))
    return serial

def random_mac() -> str:
    """Makes a MAC ID in the form expected by PPC
    
    returns:
    mac: MAC IC
    """
    mac = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return mac
