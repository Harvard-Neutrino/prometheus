from typing import Iterable, Tuple

from prometheus.particle import PropagatableParticle
from prometheus.photon_propagation.hit import Hit

def accumulate_hits(
    particles: Iterable[PropagatableParticle],
) -> Tuple[Hit, int]:
    """Makes a list of all hits for a set of particles, including
    any children. It also returns an string which identifies which
    particle produced the hit

    params
    ______
    particles: List of particles to make hits
    id_prefix: Optional string to prepend to the identifier

    returns
    _______
    hits_ids: list of tuples with hits and ids
    """

    hits_ids = []
    for _, particle in enumerate(particles):
        hits_ids += [(h, particle.serialization_idx) for h in particle.hits]
        hits_ids += accumulate_hits(particle.children)

    return hits_ids
