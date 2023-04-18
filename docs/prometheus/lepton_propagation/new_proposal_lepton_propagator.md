Module prometheus.lepton_propagation.new_proposal_lepton_propagator
===================================================================

Functions
---------

    
`init_pp_particle(particle: prometheus.particle.particle.Particle, coordinate_shift: numpy.ndarray)`
:   Initialize a PROPOSAL particle
    
    params
    ______
    particle: Prometheus particle for which to make the PROPOSAL state
    pdef: PROPOSAL particle definition
    coordinate_shift: Difference between the PROPOSAL coordinate system
        centered on the the Earth's center and Prometheus coordinate
        system in meters. The norm of this vector should be the radius
        between the center of the Earth and the start of the atmoshphere
    
    returns
    _______
    init_state: PROPOSAL particle state with energy, position, and direction
        matching input particle

    
`make_density_distributions(earth_file: str) ‑> List[~T]`
:   Make list of proposal homogeneous density distributions from
    Earth datafile
    
    params
    ______
    earth_file: data file where the parametrization of Earth is stored
    
    returns
    _______
    density_distributions: Density distributions corresponding to the 
        average density in each layer of the Earth model at linear order

    
`make_geometries(earth_file: str) ‑> List[~T]`
:   Make list of proposal geometries from earth datafile
    
    params
    ______
    earth_file: data file where the parametrization of Earth is stored
    
    returns
    _______
    geometries: List of PROPOSAL spherical shells that make up the Earth

    
`make_particle_definition(particle: prometheus.particle.particle.Particle) ‑> proposal.particle.ParticleDef`
:   Builds a proposal particle definition
    
    Parameters
    ----------
    particle: Prometheus particle you want a ParticleDef for
    
    Returns
    -------
    pdef: PROPOSAL particle definition object corresponing
        to input particle

    
`make_propagation_utilities(particle_def: proposal.particle.ParticleDef, earth_file: str, simulation_specs: dict)`
:   Make a list of PROPOSAL propagation utilities from an earth file
        for a particle given some simulation specifications
    
    params
    ______
    particle_def: PROPOSAL particle definition
    earth_file: data file where the parametrization of Earth is stored
    simulation_specs: dictionary specifying all the simulation specifications
    
    returns
    _______
    utilities: List of PROPOSAL PropagationUtility objects

    
`make_propagator(particle: prometheus.particle.particle.Particle, simulation_specs: dict, path_dict: dict) ‑> proposal.Propagator`
:   Make a PROPOSAL propagator
    
    params
    ______
    particle: Prometheus particle for which we want a PROPOSAL propagator
    simulation_specs: Dictionary specifying the configuration settings
    path_dict: Dictionary specifying any required path variables
    
    returns
    _______
    prop: PROPOSAL propagator for input Particle

    
`new_proposal_losses(prop: proposal.Propagator, particle: prometheus.particle.particle.Particle, padding: float, r_inice: float, detector_center: numpy.ndarray, coordinate_shift: numpy.ndarray) ‑> None`
:   Propagate a Prometheus particle using PROPOSAL, modifying the particle
    losses in place
    
    params
    ______
    prop: Proposal propagator porresponding to the input particle
    particle: Prometheus particle to propagate
    padding: propagation padding in meters. The propagation distance is calcuated as:
        np.linalg.norm(particle.position - detector_center) + padding
    detector_center: Center of the detector in Prometheus coordinate system in meters
    coordinate_shift: Difference between the PROPOSAL coordinate system
        centered on the the Earth's center and Prometheus coordinate
        system in meters. The norm of this vector should be the radius
        between the center of the Earth and the start of the atmoshphere, and 
        should usually only have a z-component

    
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

`NewProposalLeptonPropagator(config)`
:   Class for propagating charged leptons with PROPOSAL versions >= 7

    ### Ancestors (in MRO)

    * prometheus.lepton_propagation.lepton_propagator.LeptonPropagator

    ### Methods

    `energy_losses(self, particle: prometheus.particle.particle.Particle, detector: prometheus.detector.detector.Detector) ‑> None`
    :   Propagate a particle and track the losses. Losses and children
        are added in place
        
        params
        ______
        particle: Prometheus Particle that should be propagated
        detector: Detector that this is being propagated within
        
        returns
        _______
        propped_particle: Prometheus Particle after propagation