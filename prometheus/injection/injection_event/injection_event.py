from dataclasses import dataclass
from typing import Iterable

from .. interactions import Interactions
from ..import Particle

@dataclass
class InjectionEvent:
    initial_state: Particle
    final_states: Iterable[Particle]
    interaction: Interactions
    vertex_x: float
    vertex_y: float
    vertex_z: float
