Module prometheus.detector.module
=================================

Classes
-------

`Module(pos: numpy.ndarray, key: Tuple[int, int], noise_rate: int = 1000.0, efficiency=0.2, serial_no=None)`
:   Detector optical module
    
    Initialize a module.
    
    params
    ______
    pos: position of the optical module in meters
    key: tuple to look up module by. (string index, om index) is the convention
    [noise_rate]: noise of the module in GHz
    [efficiency]: quantum efficiency of module
    [serial number]: Serial number for the OM. I don't think you ever need to
        touch this, but I don't want to tickle a sleeping dragon