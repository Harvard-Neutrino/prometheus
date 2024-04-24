from dataclasses import dataclass
from typing import Iterable

from .injection_event import InjectionEvent

@dataclass
class SIRENInjectionEvent(InjectionEvent):
    """Injection event for SIREN injection

    params
    ______
    density_variables: the density variables of the differential cross section or decay width 
    event_weight: physical weight of the entire event calculated in SIREN
    secondary_vertex_{x,y,z}: position of any secondary interactions in the event
    """
    density_variables: Iterable[float]
    event_weight: float
    secondary_vertex_x: Iterable[float]
    secondary_vertex_y: Iterable[float]
    secondary_vertex_z: Iterable[float]
