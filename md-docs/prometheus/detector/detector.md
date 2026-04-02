Module prometheus.detector.detector
===================================

Classes
-------

`Detector(modules: List[Module], medium: Union[Medium, None])`
:   Prometheus detector object
    
    Initialize detector.
    params
    ______
    modules: List of all the modules in the detector
    medium: Medium in which the detector is embedded

    ### Instance variables

    `medium: prometheus.detector.medium.Medium`
    :

    `modules: List[prometheus.detector.module.Module]`
    :

    `n_modules: int`
    :

    `offset: numpy.ndarray`
    :

    `outer_cylinder: Tuple[float, float]`
    :

    `outer_radius: float`
    :

    ### Methods

    `to_f2k(self, geo_file: str, serial_nos: List[str] = [], mac_ids: List[str] = []) ‑> None`
    :   Write detector corrdinates into f2k format.
        
        params
        ______
        geo_file: file name of where to write it
        serial_nos: serial numbers for the optical modules. These MUST be in hexadecimal
            format, but there exact value does not matter. If nothing is provided, these
            values will be randomly generated
        mac_ids: MAC (I don't think this is actually what this is called) IDs for the DOMs.
            By default these will be randomly generated. This is prbably what you want
            to do.
        
        raises
        ______

`IncompatibleMACIDsError()`
:   Raised when MAC IDs length doesn't match number of DOMs

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

`IncompatibleSerialNumbersError()`
:   Raised when serial numbers length doesn't match number of DOMs

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException