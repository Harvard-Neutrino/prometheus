from dataclasses import dataclass

from .injection_event import InjectionEvent

@dataclass
class LIInjectionEvent(InjectionEvent):
    bjorken_x: float
    bjorken_y: float
    column_depth: float
