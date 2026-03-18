import numpy as np
import proposal as pp
from .particle import PropagatableParticle
from ..utils.units import cm_to_m, MeV_to_GeV

def particle_from_proposal(
    pp_particle, 
    coordinate_offset,
    parent: PropagatableParticle = None,
) -> PropagatableParticle:
    """Create a Prometheus particle from a PROPOSAL object.

    Parameters
    ----------
    pp_particle
        PROPOSAL particle instance.
    coordinate_offset : numpy.ndarray
        Coordinate offset to subtract from the particle position.
    parent : PropagatableParticle, optional
        Parent particle that created this particle.

    Returns
    -------
    child : PropagatableParticle
        New particle which is a child of ``parent``.
    """
    pdg_code = pp_particle.type
    e = pp_particle.energy * MeV_to_GeV
    position = np.array(
        [pp_particle.position.x, pp_particle.position.y, pp_particle.position.z]
    ) * cm_to_m - coordinate_offset
    direction = np.array(
        [pp_particle.direction.x, pp_particle.direction.y, pp_particle.direction.z]
    )
    child = PropagatableParticle(pdg_code, e, position, direction, None, parent)
    return child
