from dataclasses import dataclass

from .injection_event import InjectionEvent

@dataclass
class LIInjectionEvent(InjectionEvent):
    """Injection event for LeptonInjector injection

    params
    ______
    bjorken_x: Bjorken-x variable of the interaction, i.e. fraction
        of the incident neutrino momentum imparted to the hadronic
        shower
    bjorken_y: Bjorken-y variable of the interaction, i.e. fraction
        of the incident neutrino energy imparted to the hadronic
        shower
    column_depth: column depth traversed by the neutrino before
        interacting in M.W.E.
    """
    bjorken_x: float
    bjorken_y: float
    column_depth: float
