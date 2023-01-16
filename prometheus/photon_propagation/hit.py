from dataclasses import dataclass
from typing import Optional

@dataclass
class Hit:
    """Dataclass for tracking a OM seeing light"""
    string_id: int
    om_id: int
    time: float # ns
    wavelength: Optional[float] # nm
    om_zenith: Optional[float] # radian
    om_azimuth: Optional[float] # radian
    photon_zenith: Optional[float] # radian
    photon_azimuth: Optional[float] # radian
