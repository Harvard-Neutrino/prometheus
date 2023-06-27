Module prometheus.photon_propagation.ppc_photon_propagator
==========================================================

Functions
---------

    
`ppc_sim(particle: prometheus.particle.particle.Particle, det: prometheus.detector.detector.Detector, lp: prometheus.lepton_propagation.lepton_propagator.LeptonPropagator, ppc_config: dict) ‑> None`
:   Simulate the propagation of a particle and of any photons resulting from
    the energy losses of this particle
    
    params
    ______
    particle: Particle to propagate
    det: Detector object to simulate within
    lp: Prometheus LeptonPropagator to imulate any charged leptons
    ppc_config: dictionary containg the configuration settings for the photon propagation

Classes
-------

`PPCPhotonPropagator(lepton_propagator: prometheus.lepton_propagation.lepton_propagator.LeptonPropagator, detector: prometheus.detector.detector.Detector, photon_prop_config: dict)`
:   Interface for simulating energy losses and light propagation using PPC
    
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

    `propagate(self, particle: prometheus.particle.particle.Particle) ‑> None`
    :   Propagate input particle using PPC. Instead it modifies the 
        state of the input Particle. We should make this more consistent 
        but that is a problem for another day...
        
        params
        ______
        particle: Prometheus particle to propagate