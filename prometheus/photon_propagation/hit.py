from dataclasses import dataclass
from typing import Optional

@dataclass
class Hit:
    """Dataclass for tracking an OM seeing light.

    Attributes
    ----------
    string_id : int
        String index of the OM that was hit.
    om_id : int
        OM index of the OM that was hit.
    time : float
        Time of photon arrival in ns.
    wavelength : float, optional
        Photon wavelength in nm.
    om_zenith : float, optional
        Zenith angle that the photon arrived at on the OM in radians.
    om_azimuth : float, optional
        Azimuthal angle that the photon arrived at on the OM in radians.
    photon_zenith : float, optional
        Zenith angle of the photon momentum in radians.
    photon_azimuth : float, optional
        Azimuthal angle of the photon momentum in radians.
    """
    string_id: int
    om_id: int
    time: float # ns
    wavelength: Optional[float] # nm
    om_zenith: Optional[float] # radian
    om_azimuth: Optional[float] # radian
    photon_zenith: Optional[float] # radian
    photon_azimuth: Optional[float] # radian
