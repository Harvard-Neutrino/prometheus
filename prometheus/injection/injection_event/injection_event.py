from dataclasses import dataclass
from typing import Iterable

from .. interactions import Interactions
from ..import Particle

# TODO Why is vertex split up ? DTaSD
@dataclass
class InjectionEvent:
    """Dataclass for handling injection events.

    Parameters
    ----------
    initial_state : Particle
        Incident neutrino.
    final_states : Iterable[Particle]
        All particles that result from the interaction.
    interaction : Interactions
        Type of interaction.
    vertex_x : float
        X-position of the interaction vertex.
    vertex_y : float
        Y-position of the interaction vertex.
    vertex_z : float
        Z-position of the interaction vertex.
    """
    initial_state: Particle
    final_states: Iterable[Particle]
    interaction: Interactions
    vertex_x: float
    vertex_y: float
    vertex_z: float
