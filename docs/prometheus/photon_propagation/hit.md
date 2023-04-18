Module prometheus.photon_propagation.hit
========================================

Classes
-------

`Hit(string_id: int, om_id: int, time: float, wavelength: Optional[float], om_zenith: Optional[float], om_azimuth: Optional[float], photon_zenith: Optional[float], photon_azimuth: Optional[float])`
:   Dataclass for tracking a OM seeing light
    
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

    ### Class variables

    `om_azimuth: Optional[float]`
    :

    `om_id: int`
    :

    `om_zenith: Optional[float]`
    :

    `photon_azimuth: Optional[float]`
    :

    `photon_zenith: Optional[float]`
    :

    `string_id: int`
    :

    `time: float`
    :

    `wavelength: Optional[float]`
    :