import numpy as np
import awkward as ak
from typing import Callable

from .. import Particle, PropagatableParticle
from ..injection_event import LIInjectionEvent
from ..interactions import Interactions
from .injection import Injection
from .LI_injection import LIInjection

def prometheus_inj_to_li_injection_event(truth: ak.Record) -> LIInjectionEvent:
    initial_state = Particle(
        truth["initial_type"],
        truth["initial_energy"],
        np.array([
            truth["initial_x"],
            truth["initial_y"],
            truth["initial_z"]
        ]),
        np.array([
            np.cos(truth["initial_azimuth"]) * np.sin(truth["initial_zenith"]),
            np.sin(truth["initial_azimuth"]) * np.sin(truth["initial_zenith"]),
            np.cos(truth["initial_zenith"]),
        ])
    )
    final_states = []
    for idx, parent in enumerate(truth["parent"]):
        if parent != 0:
            continue
        final_state = PropagatableParticle(
            truth["final_state_type", idx],
            truth["final_state_energy", idx],
            np.array([
                truth["final_state_x", idx],
                truth["final_state_y", idx],
                truth["final_state_z", idx]
            ]),
            np.array([
                np.cos(truth["final_state_azimuth", idx]) * np.sin(truth["final_state_zenith", idx]),
                np.sin(truth["final_state_azimuth", idx]) * np.sin(truth["final_state_zenith", idx]),
                np.cos(truth["final_state_zenith", idx]),
            ]),
            initial_state
        )
        final_states.append(final_state)
    interaction = Interactions(truth["interaction"])

    return LIInjectionEvent(
        initial_state,
        final_states,
        interaction,
        truth["initial_x"],
        truth["initial_y"],
        truth["initial_z"],
        truth["bjorken_x"],
        truth["bjorken_y"],
        truth["column_depth"]
    )

def injection_from_prometheus(
        file: str,
        injection_cls: Injection = LIInjection,
        event_converter: Callable = prometheus_inj_to_li_injection_event
    ):
        """Make an Injection object from Prometheus output. If not using
        output that was not generated from LeptonInjector, you will need
        to tell this which kind of injection you want to use, and how to
        convert each item to the appropriate InjectionEvent
        """
        a = ak.from_parquet(file)
        events = [event_converter(truth) for truth in a["mc_truth"]]
        return injection_cls(events)

