# -*- coding: utf-8 -*-
# injection.py
# Copyright (C) 2022 Jeffrey Lazar, Stephan Meighen-Berger
# Interface class to the different lepton injectors

import numpy as np
import awkward as ak
from typing import Iterable, Union

from .. import Particle
from ..injection_event.injection_event import InjectionEvent

def recursive_getattr(x, attr):
    for a in attr.split("."):
        x = getattr(x, a)
    return x

def recursively_get_final_property(
    particles: Iterable[Particle],
    attr: str,
    idx = None
) -> Iterable[Union[float, int]]:
    l = np.array([])
    for particle in particles:
        if idx is None:
            l = np.hstack([l, recursive_getattr(particle, attr)])
        else:
            l = np.hstack([l, recursive_getattr(particle, attr)[idx]])
        l = np.hstack([
            l, recursively_get_final_property(particle.children, attr, idx=idx)
        ])
    return l

class Injection:
    
    def __init__(
        self,
        events: Iterable[InjectionEvent]
    ):
        self._events = events
        self._size = len(events)
        self._current_idx = 0

    def __getitem__(self, idx):
        return self.events[idx]

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx >= self._size:
            self._current_idx = 0
            raise StopIteration
        event = self.events[self._current_idx]
        self._current_idx += 1
        return event

    @property
    def events(self):
        return self._events

    def to_dict(self) -> dict:
        # TODO make this recursive
        d = {}
        d["interaction"] = [x.interaction.value for x in self]
        d["initial_energy"] = [x.initial_state.e for x in self]
        d["initial_type"] = [x.initial_state.pdg_code for x in self]
        d["initial_zenith"] = [x.initial_state.theta for x in self]
        d["initial_azimuth"] = [x.initial_state.phi for x in self]
        d["initial_x"] = [x.initial_state.position[0] for x in self]
        d["initial_y"] = [x.initial_state.position[1] for x in self]
        d["initial_z"] = [x.initial_state.position[2] for x in self]
        final_state_es = []
        final_state_types = []
        final_state_zeniths = []
        final_state_azimuths = []
        final_state_xs = []
        final_state_ys = []
        final_state_zs = []
        parents = []
        for event in self:
            final_state_es.append(recursively_get_final_property(event.final_states, "e"))
            final_state_types.append(recursively_get_final_property(event.final_states, "pdg_code"))
            final_state_zeniths.append(recursively_get_final_property(event.final_states, "theta"))
            final_state_azimuths.append(recursively_get_final_property(event.final_states, "phi"))
            final_state_xs.append(recursively_get_final_property(event.final_states, "position", 0))
            final_state_ys.append(recursively_get_final_property(event.final_states, "position", 1))
            final_state_zs.append(recursively_get_final_property(event.final_states, "position", 2))
            parents.append(recursively_get_final_property(event.final_states, "parent.serialization_idx"))
        d["final_state_energy"] = final_state_es
        d["final_state_type"] = final_state_types
        d["final_state_zenith"] = final_state_zeniths
        d["final_state_azimuth"] = final_state_azimuths
        d["final_state_x"] = final_state_xs
        d["final_state_y"] = final_state_ys
        d["final_state_z"] = final_state_zs
        d["parent"] = parents

        return d

    def to_awkward(self) -> ak.Array:
        return ak.Array(self.to_dict())
