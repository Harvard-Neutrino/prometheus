import numpy as np

from ..utils.units import MeV_to_GeV, cm_to_m, s_to_ns
from .particle import PropagatableParticle


def particle_from_proposal(
    pp_particle,
    coordinate_offset,
    parent: PropagatableParticle,
) -> PropagatableParticle:
    """Create a Prometheus particle from a PROPOSAL object.

    Parameters
    ----------
    pp_particle : object
        PROPOSAL particle instance.
    coordinate_offset : np.ndarray
        Coordinate offset to subtract from the particle position.
    parent : PropagatableParticle
        Parent particle that created this particle. Its start time anchors
        the child's time to the event start.

    Returns
    -------
    child : PropagatableParticle
        New particle which is a child of ``parent``.
    """
    pdg_code = pp_particle.type
    e = pp_particle.energy * MeV_to_GeV
    position = (
        np.array([pp_particle.position.x, pp_particle.position.y, pp_particle.position.z]) * cm_to_m
        - coordinate_offset
    )
    direction = np.array(
        [pp_particle.direction.x, pp_particle.direction.y, pp_particle.direction.z]
    )
    # PROPOSAL times are in seconds and relative to that propagation's own
    # start (each propagation restarts its clock at 0), so pp_particle.time is
    # only the parent's flight time to this vertex. The parent's start time
    # must be added to anchor the child to the event start, as the Particle
    # time convention requires; without it, chains deeper than one generation
    # (e.g. tau -> mu -> decay) are timed early by the ancestors' flight time.
    time = parent.time + pp_particle.time * s_to_ns
    child = PropagatableParticle(pdg_code, e, position, direction, None, parent, time=time)
    return child
