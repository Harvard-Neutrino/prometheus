Module prometheus.photon_propagation.olympus_photon_propagator
==============================================================

Classes
-------

`OlympusPhotonPropagator(lepton_propagator: prometheus.lepton_propagation.lepton_propagator.LeptonPropagator, detector: prometheus.detector.detector.Detector, config: dict)`
:   PhotonPropagator the uses Olympus to propagate photons
    
    Initialize the PhotonPropagator object
    
    params
    ______
    lepton_propagator: Prometheus LeptonPropagator object which will be used
        to generate losses from the particles
    detector: Prometheus detector object in which the light will be
        propagated
    photon_prop_config: dictionary containing all the configuration settings
        for photon propagation

    ### Ancestors (in MRO)

    * prometheus.photon_propagation.photon_propagator.PhotonPropagator

    ### Methods

    `propagate(self, particle: prometheus.particle.particle.Particle)`
    :   Simulate losses and propagate resulting photons for input particle
        
        params
        ______
        particle: Prometheus Particle object to simulate
        
        returns
        _______
        res_event: PLEASE FILL THIS IN
        res_record: PLEASE FILL THIS IN