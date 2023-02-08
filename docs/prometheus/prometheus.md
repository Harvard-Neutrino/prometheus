Module prometheus.prometheus
============================

Functions
---------

    
`regularize(s: str) ‑> str`
:   Helper fnuction to regularize strings
    
    params
    ______
    s: string to regularize
    
    returns
    _______
    s: regularized string

Classes
-------

`Prometheus(userconfig: Union[None, dict, str] = None, detector: Optional[None] = None)`
:   Class for unifying injection, energy loss calculation, and photon propagation
    
    Initializes the Prometheus class
    
    params
    ______
    userconfig: Configuration dictionary or path to yaml file 
        which specifies configuration
    detector: Detector to be used or path to geo file to load detector file.
        If this is left out, the path from the `userconfig["detector"]["specs file"]`
        be loaded
    
    raises
    ______
    UnknownInjectorError: If we don't know how to handle the injector the config
        is asking for
    UnknownLeptonPropagatorError: If we don't know how to handle the lepton
        propagator you are asking for
    UnknownPhotonPropagatorError: If we don't know how to handle the photon
        propagator you are asking for
    CannotLoadDetectorError: When no detector provided and no
        geo file path provided in config

    ### Instance variables

    `detector`
    :

    `injection`
    :

    ### Methods

    `construct_output(self)`
    :   Constructs a parquet file with metadata from the generated files.
        Currently this still treats olympus and ppc output differently.

    `inject(self)`
    :   Determines initial neutrino and final particle states according to config

    `propagate(self)`
    :   Calculates energy losses, generates photon yields, and propagates photons

    `sim(self)`
    :   Performs injection of precipitating interaction, calculates energy losses,
        calculates photon yield, propagates photons, and save resultign photons