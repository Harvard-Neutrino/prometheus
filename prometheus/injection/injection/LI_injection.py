import numpy as np
import h5py as h5
from typing import Iterable

from .. import Particle, PropagatableParticle
from ..injection_event import LIInjectionEvent
from ..interactions import Interactions
from .injection import Injection

class LIInjection(Injection):

    def __init__(self, events: Iterable[LIInjectionEvent]):
        if not all([isinstance(event, LIInjectionEvent) for event in events]):
            raise ValueError("You are trying to make LI Injection with non-LI events")
        super().__init__(events)
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d["bjorken_x"] = [x.bjorken_x for x in self]
        d["bjorken_y"] = [x.bjorken_y for x in self]
        d["column_depth"] = [x.column_depth for x in self]

        return d

def injection_from_LI_output(LI_file: str) -> LIInjection:
    """Creates injection object from a saved LI file"""
    with h5.File(LI_file, "r") as h5f:
        injectors = list(h5f.keys())
        if len(injectors) > 1:
            raise ValueError("Too many injectors")
        injection = h5f[injectors[0]]
        injection_events = [
            injection_event_from_LI(injection, idx)
            for idx in range(injection["initial"].shape[0])
        ]
        return LIInjection(injection_events)

INTERACTION_CONVERTER = {
    (12, -2000001006, 11): Interactions.CHARGED_CURRENT,
    (14, -2000001006, 13): Interactions.CHARGED_CURRENT,
    (16, -2000001006, 15): Interactions.CHARGED_CURRENT,
    (12, 11, -2000001006): Interactions.CHARGED_CURRENT,
    (14, 13, -2000001006): Interactions.CHARGED_CURRENT,
    (16, 15, -2000001006): Interactions.CHARGED_CURRENT,
    (-12, -2000001006, -11): Interactions.CHARGED_CURRENT,
    (-14, -2000001006, -13): Interactions.CHARGED_CURRENT,
    (-16, -2000001006, -15): Interactions.CHARGED_CURRENT,
    (-12, -11, -2000001006): Interactions.CHARGED_CURRENT,
    (-14, -13, -2000001006): Interactions.CHARGED_CURRENT,
    (-16, -15, -2000001006): Interactions.CHARGED_CURRENT,
    (12, 12, -2000001006): Interactions.NEUTRAL_CURRENT,
    (14, 14, -2000001006): Interactions.NEUTRAL_CURRENT,
    (16, 16, -2000001006): Interactions.NEUTRAL_CURRENT,
    (12, -2000001006, 12): Interactions.NEUTRAL_CURRENT,
    (14, -2000001006, 14): Interactions.NEUTRAL_CURRENT,
    (16, -2000001006, 16): Interactions.NEUTRAL_CURRENT,
    (-12, -12,-2000001006): Interactions.NEUTRAL_CURRENT,
    (-14, -14,-2000001006): Interactions.NEUTRAL_CURRENT,
    (-16, -16,-2000001006): Interactions.NEUTRAL_CURRENT,
    (-12,-2000001006, -12): Interactions.NEUTRAL_CURRENT,
    (-14,-2000001006, -14): Interactions.NEUTRAL_CURRENT,
    (-16,-2000001006, -16): Interactions.NEUTRAL_CURRENT,
    (-12, -2000001006, -2000001006): Interactions.GLASHOW_RESONANCE,
    (-12,-12, 11): Interactions.GLASHOW_RESONANCE,
    (-12,-14, 13): Interactions.GLASHOW_RESONANCE,
    (-12,-16, 15): Interactions.GLASHOW_RESONANCE,
    (-12, 11,-12): Interactions.GLASHOW_RESONANCE,
    (-12, 13,-14): Interactions.GLASHOW_RESONANCE,
    (-12, 15,-16): Interactions.GLASHOW_RESONANCE,
    (14, 13, -13): Interactions.DIMUON,
    (14, -13, 13): Interactions.DIMUON,
    (-14, -13, 13): Interactions.DIMUON,
    (-14, 13, -13): Interactions.DIMUON,
}

def injection_event_from_LI(injection: h5.Group, idx: int) -> LIInjectionEvent:
    """Create an injection event from LI h5 group and index

    params
    ______
    injection: Group from h5 file to make injection from
    idx: index in that gorup to make event

    returns
    _______
    event: Prometheus LIInjectionEvent corresponding to input
    """
    direction = injection["initial"]["Direction"][idx]
    theta = direction[0]
    phi = direction[1]
    initial_state = Particle(
        injection["properties"]["initialType"][idx],
        injection["initial"]["Energy"][idx],
        injection["initial"]["Position"][idx],
        np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]),
        None
    )
    final_states = []
    for final_ctr in [1,2]:
        direction = injection[f"final_{final_ctr}"]["Direction"][idx]
        theta = direction[0]
        phi = direction[1]
        final_state = PropagatableParticle(
            injection["properties"][f"finalType{final_ctr}"][idx],
            injection[f"final_{final_ctr}"]["Energy"][idx],
            injection[f"final_{final_ctr}"]["Position"][idx],
            np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]),
            None,
            initial_state
        )
        final_states.append(final_state)
    interaction = INTERACTION_CONVERTER[(
        initial_state.pdg_code,
        final_states[0].pdg_code,
        final_states[1].pdg_code,
    )]
    vertex_x = injection["properties"]["x"][idx]
    vertex_y = injection["properties"]["y"][idx]
    vertex_z = injection["properties"]["z"][idx]
    bjorken_x = injection["properties"]["finalStateX"][idx]
    bjorken_y = injection["properties"]["finalStateY"][idx]
    column_depth = injection["properties"]["totalColumnDepth"][idx]
    
    event = LIInjectionEvent(
        initial_state,
        final_states,
        interaction,
        vertex_x,
        vertex_y,
        vertex_z,
        bjorken_x,
        bjorken_y,
        column_depth
    )
    return event
