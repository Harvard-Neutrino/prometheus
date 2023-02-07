from dataclasses import dataclass
from typing import Optional

@dataclass
class Hit:
    """Dataclass for tracking a OM seeing light
    
    fields
    ______
    string_id: String index of the OM that was hit
    om_id: OM index of the OM that was hit
    time: time of photon arrival in ns
    wavelength: photon wavelength in nm
    om_zenith: zenith angle that the photon arrived at on the OM
    om_azimuth: azimuthal angle that the photon arrived at
    photon_zenith: zenith angle of the photon momentum
    photon_azimuth: azimuthal angle of the photon momentum
    """
    string_id: int
    om_id: int
    time: float # ns
    wavelength: Optional[float] # nm
    om_zenith: Optional[float] # radian
    om_azimuth: Optional[float] # radian
    photon_zenith: Optional[float] # radian
    photon_azimuth: Optional[float] # radian
