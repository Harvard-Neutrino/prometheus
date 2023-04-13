Module prometheus.particle.particle
===================================

Classes
-------

`Particle(pdg_code: int, e: float, position: numpy.ndarray, direction: numpy.ndarray, serialization_idx: int)`
:   Base dataclass for particle event structure
    
    fields
    ______
    pdg_code: PDG mc code
    e: energy in GeV
    position: particle position in meters
    direction: unit vector pointing along particle momentum
    serialization_idx: Index helper for serialization. This
        will be overwritten at serialization time

    ### Descendants

    * prometheus.particle.particle.PropagatableParticle

    ### Class variables

    `direction: numpy.ndarray`
    :

    `e: float`
    :

    `pdg_code: int`
    :

    `position: numpy.ndarray`
    :

    `serialization_idx: int`
    :

    ### Instance variables

    `phi`
    :

    `theta`
    :

`PropagatableParticle(pdg_code: int, e: float, position: numpy.ndarray, direction: numpy.ndarray, serialization_idx: int, parent: prometheus.particle.particle.Particle, children: List[prometheus.particle.particle.Particle] = <factory>, losses: List[~T] = <factory>, hits: List[~T] = <factory>)`
:   Particle event structure with added feature to support propagation
    
    fields
    ______
    parent: Particle which created this particle
    children: Particles that this one spawned
    losses: energy losses created by this particle
    hits: OM hits originating from this particle

    ### Ancestors (in MRO)

    * prometheus.particle.particle.Particle

    ### Class variables

    `children: List[prometheus.particle.particle.Particle]`
    :

    `hits: List[~T]`
    :

    `losses: List[~T]`
    :

    `parent: prometheus.particle.particle.Particle`
    :