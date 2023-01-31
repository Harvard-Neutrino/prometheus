from typing import Iterable

from prometheus.particle import Particle
from prometheus.injection import Injection

def recursion_helper(particles: Iterable[Particle], idx0: int) -> int:
    idx = idx0
    for particle in particles:
        particle.serialization_idx = idx
        idx = recursion_helper(particle.children, idx+1)
    return idx

def set_serialization_index(injection: Injection) -> None:
    for injection_event in injection:
        injection_event.initial_state.serialization_idx = 0
        recursion_helper(injection_event.final_states, 1)
