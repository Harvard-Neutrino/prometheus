Module prometheus.injection.lepton_injector_utils
=================================================

Functions
---------

    
`apply_detector_offset(injection_file: str, detector_offset: numpy.ndarray) ‑> None`
:   Translate the injection to a detector-centered coordinate system
    
    params
    ______
    injection_file: File where the untranslated injection is saved
    detector_offset: Center of the detector in meters

    
`make_new_LI_injection(path_dict: dict, injection_specs: dict, detector_offset: numpy.ndarray) ‑> None`
:   Make a new injection with LeptonInjector
    
    params
    ______
    path_dict: dictionary specifying all the necessary pathing information
    injection_specs: dictionary specifying all the injection configuration
        settings
    detector_offset: Center of the detector in meters