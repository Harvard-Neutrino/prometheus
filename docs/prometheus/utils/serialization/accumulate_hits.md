Module prometheus.utils.serialization.accumulate_hits
=====================================================

Functions
---------

    
`accumulate_hits(particles: Iterable[prometheus.particle.particle.PropagatableParticle]) ‑> Tuple[prometheus.photon_propagation.hit.Hit, int]`
:   Makes a list of all hits for a set of particles, including
    any children. It also returns an string which identifies which
    particle produced the hit
    
    params
    ______
    particles: List of particles to make hits
    id_prefix: Optional string to prepend to the identifier
    
    returns
    _______
    hits_ids: list of tuples with hits and ids