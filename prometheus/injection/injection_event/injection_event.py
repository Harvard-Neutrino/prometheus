from dataclasses import dataclass
from typing import Iterable

from .. interactions import Interactions
from ..import Particle

# TODO Why is vertex split up ? DTaSD
@dataclass
class InjectionEvent:
    """Dataclass for handing injection events
    
    params
    ______
    initial_state: Incident neutrino
    final_states: All particles which results from the interaction
    vertex_x: x-position of the interaction vertex
    vertex_y: x-position of the interaction vertex
    vertex_z: x-position of the interaction vertex
    """
    initial_state: Particle
    final_states: Iterable[Particle]
    interaction: Interactions
    vertex_x: float
    vertex_y: float
    vertex_z: float
