Module prometheus.photon_propagation.photon_propagator
======================================================

Classes
-------

`PhotonPropagator(lepton_propagator: prometheus.lepton_propagation.lepton_propagator.LeptonPropagator, detector: prometheus.detector.detector.Detector, photon_prop_config: dict)`
:   Interface for handling different photon propagators
    
    Initialize the PhotonPropagator object
    
    params
    ______
    lepton_propagator: Prometheus LeptonPropagator object which will be used
        to generate losses from the particles
    detector: Prometheus detector object in which the light will be
        propagated
    photon_prop_config: dictionary containing all the configuration settings
        for photon propagation

    ### Descendants

    * prometheus.photon_propagation.olympus_photon_propagator.OlympusPhotonPropagator
    * prometheus.photon_propagation.ppc_photon_propagator.PPCPhotonPropagator

    ### Instance variables

    `config: dict`
    :

    `detector: prometheus.detector.detector.Detector`
    :

    `lepton_propagator: prometheus.lepton_propagation.lepton_propagator.LeptonPropagator`
    :

    ### Methods

    `propagate(self, particle)`
    :