from typing import Iterable

import h5py as h5
import numpy as np

from .. import Particle, PropagatableParticle
from ..injection_event import LIInjectionEvent
from ..interactions import INTERACTION_CONVERTER
from .injection import Injection


class LIInjection(Injection):
    """Injection constructed from LeptonInjector output."""

    def __init__(self, events: Iterable[LIInjectionEvent]):
        if not all([isinstance(event, LIInjectionEvent) for event in events]):
            raise ValueError("You are trying to make LI Injection with non-LI events")
        super().__init__(events)

    def to_dict(self) -> dict:
        """Convert all properties of the injection to a dictionary.

        Extends the base implementation with LeptonInjector-specific fields
        (Bjorken x, Bjorken y, and column depth).

        Returns
        -------
        d : dict
            Dictionary with all injection properties.
        """
        d = super().to_dict()
        d["bjorken_x"] = [x.bjorken_x for x in self]
        d["bjorken_y"] = [x.bjorken_y for x in self]
        d["column_depth"] = [x.column_depth for x in self]

        return d


def injection_from_LI_output(LI_file: str) -> LIInjection:
    """Create an injection object from a saved LI file.

    Parameters
    ----------
    LI_file : str
        Path to the LeptonInjector output file.

    Returns
    -------
    LIInjection
        Injection object constructed from the LI file contents.
    """
    with h5.File(LI_file, "r") as h5f:
        injectors = list(h5f.keys())
        if len(injectors) > 1:
            raise ValueError("Too many injectors")
        injection = h5f[injectors[0]]
        injection_events = [
            injection_event_from_LI(injection, idx) for idx in range(injection["initial"].shape[0])
        ]
        return LIInjection(injection_events)


def injection_event_from_LI(injection: h5.Group, idx: int) -> LIInjectionEvent:
    """Create an injection event from an LI H5 group and index.

    Parameters
    ----------
    injection : h5.Group
        Group from the H5 file to create the injection from.
    idx : int
        Index in that group to create the event from.

    Returns
    -------
    event : LIInjectionEvent
        Prometheus LIInjectionEvent corresponding to input.
    """
    direction = injection["initial"]["Direction"][idx]
    theta = direction[0]
    phi = direction[1]
    initial_state = Particle(
        injection["properties"]["initialType"][idx],
        injection["initial"]["Energy"][idx],
        injection["initial"]["Position"][idx],
        np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        ),
        None,
    )
    final_states = []
    for final_ctr in [1, 2]:
        direction = injection[f"final_{final_ctr}"]["Direction"][idx]
        theta = direction[0]
        phi = direction[1]
        final_state = PropagatableParticle(
            injection["properties"][f"finalType{final_ctr}"][idx],
            injection[f"final_{final_ctr}"]["Energy"][idx],
            injection[f"final_{final_ctr}"]["Position"][idx],
            np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            ),
            None,
            initial_state,
        )
        final_states.append(final_state)
    interaction = INTERACTION_CONVERTER[
        (
            initial_state.pdg_code,
            final_states[0].pdg_code,
            final_states[1].pdg_code,
        )
    ]
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
        column_depth,
    )
    return event
