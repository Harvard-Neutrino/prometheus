Module prometheus.lepton_propagation.old_proposal_lepton_propagator
===================================================================

Functions
---------

    
`init_dynamic_data(particle: prometheus.particle.particle.Particle, particle_definition: proposal.particle.ParticleDef, coordinate_shift: numpy.ndarray) ‑> proposal.particle.DynamicData`
:   Makes PROPOSAL DynamicData:
    
    params
    ______
    particle: Prometheus you want DynamicData for
    particle_definition: PROPOSAL particle definition
    
    returns
    _______
    particle_dd: PROPOSAL DynamicData for input particle

    
`make_detector(earth_file: str) ‑> proposal.geometry.Sphere`
:   Make a PROPOSAL sphere with radius eqaul to the max radius
    from an Earth data file
    
    params
    ______
    earth_file: Earth datafile
    
    returns
    _______
    detector: PROPOSAL Sphere

    
`make_particle_def(particle: prometheus.particle.particle.Particle) ‑> proposal.particle.ParticleDef`
:   Makes a PROPOSAL particle definition
    
    params
    ______
    particle: Prometheus particle for which we want a PROPOSAL ParticleDef
    
    returns
    _______
    pdef: PROPOSAL particle definition

    
`make_propagator(particle: str, simulation_specs: dict, path_dict: dict) ‑> proposal.Propagator`
:   Make a PROPOSAL propagator
    
    params
    ______
    particle: Prometheus particle for which we want a PROPOSAL propagator
    simulation_specs: Dictionary specifying the configuration settings
    path_dict: Dictionary specifying any required path variables
    
    returns
    _______
    prop: PROPOSAL propagator for input Particle

    
`make_sector_defs(earth_file: str, simulation_specs: dict) ‑> List[proposal.SectorDefinition]`
:   Make list of PROPOSAL sector definitions for an input Earth 
    data file according to given simulation specifications
    
    params
    ______
    earth_file: Earth datafile
    simulation_specs: dictionary specifying simulation paramters
    
    returns
    _______
    sec_defs: list of PROPOSAL sector definitions

    
`old_proposal_losses(prop: proposal.Propagator, pdef: proposal.particle.ParticleDef, particle: prometheus.particle.particle.Particle, padding: float, r_inice: float, detector_center: numpy.ndarray, coordinate_shift: numpy.ndarray) ‑> prometheus.particle.particle.Particle`
:   Propagates charged lepton using PROPOSAL version <= 6
    
    params
    ______
    prop: PROPOSAL propagator object for the charged lepton to be propagated
    pdef: PROPOSAL particle definition for the charged lepton
    particle: Prometheus particle object to be propagated
    padding: Distance to propagate the charged lepton beyond its distance from the
        center of the detector
    r_inice: Distance from the center of the edge detector where losses should be
        recorded. This should be a few scattering lengths for accuracy, but not too
        much more because then you will propagate light which never makes it
    detector_center: Center of the detector in meters

    
`remove_comments(s: str) ‑> str`
:   Helper for removing trailing comments
    
    params
    ______
    s: string you want to remove comments from
    
    returns
    _______
    s: string without the comments

Classes
-------

`OldProposalLeptonPropagator(config: dict)`
:   Class for propagating charged leptons with PROPOSAL versions <= 6

    ### Ancestors (in MRO)

    * prometheus.lepton_propagation.lepton_propagator.LeptonPropagator

    ### Methods

    `energy_losses(self, particle: prometheus.particle.particle.Particle, detector: prometheus.detector.detector.Detector) ‑> None`
    :   Propagate a particle and track the losses. Losses and 
        children are applied in place
        
        params
        ______
        particle: Prometheus Particle that should be propagated
        detector: Detector that this is being propagated within
            This is a temporary fix and will hopefully we solved soon :-)
        
        returns
        _______
        propped_particle: Prometheus Particle after propagation