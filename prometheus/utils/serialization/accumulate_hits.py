from typing import Iterable, Tuple

from prometheus.particle import PropagatableParticle
from prometheus.photon_propagation.hit import Hit

def accumulate_hits(
    particles: Iterable[PropagatableParticle],
) -> Tuple[Hit, int]:
    """Create a list of hits for a set of particles, including their children.

    Parameters
    ----------
    particles : Iterable[PropagatableParticle]
        List of particles to collect hits for.

    Returns
    -------
    list of tuple[Hit, int]
        List of tuples ``(hit, serialization_idx)`` identifying which
        particle produced each hit.
    """

    hits_ids = []
    for _, particle in enumerate(particles):
        hits_ids += [(h, particle.serialization_idx) for h in particle.hits]
        hits_ids += accumulate_hits(particle.children)

    return hits_ids
