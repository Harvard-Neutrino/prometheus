from typing import Iterable

from prometheus.particle import Particle
from prometheus.injection import Injection

def recursion_helper(particles: Iterable[Particle], idx0: int) -> int:
    """Recursively assign serialization indices to a tree of particles.

    Parameters
    ----------
    particles : Iterable[Particle]
        Iterable of particles at the current level of the tree.
    idx0 : int
        Index to assign to the first particle in this batch.

    Returns
    -------
    idx : int
        The next available index after processing all particles and their children.
    """
    idx = idx0
    for particle in particles:
        particle.serialization_idx = idx
        idx = recursion_helper(particle.children, idx+1)
    return idx

def set_serialization_index(injection: Injection) -> None:
    """Assign a unique serialization index to every particle in the injection.

    Parameters
    ----------
    injection : Injection
        Prometheus injection whose particle trees are to be indexed.
    """
    for injection_event in injection:
        injection_event.initial_state.serialization_idx = 0
        recursion_helper(injection_event.final_states, 1)
