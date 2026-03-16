from dataclasses import dataclass

from .injection_event import InjectionEvent

@dataclass
class LIInjectionEvent(InjectionEvent):
    """Injection event for `LeptonInjector` injection.

    Parameters
    ----------
    bjorken_x : float
        Bjorken-x variable of the interaction, i.e. fraction of the
        incident neutrino momentum imparted to the hadronic shower.
    bjorken_y : float
        Bjorken-y variable of the interaction.
    column_depth : float
        Column depth traversed by the neutrino before interacting in
        M.W.E.
    """
    bjorken_x: float
    bjorken_y: float
    column_depth: float
