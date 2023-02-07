import numpy as np
import proposal as pp
from .particle import PropagatableParticle
from ..utils.units import cm_to_m, MeV_to_GeV

def particle_from_proposal(
    pp_particle, 
    coordinate_offset,
    parent: PropagatableParticle = None,
) -> PropagatableParticle:
    """Creates a Prometheus particle from a PROPOSAL object

    params
    ______
    pp_particle: PROPOSAL particle

    returns
    _______
    child: the new particle which is a child of parent
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
