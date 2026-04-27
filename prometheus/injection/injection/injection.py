# -*- coding: utf-8 -*-
# injection.py
# Copyright (C) 2022 Jeffrey Lazar, Stephan Meighen-Berger
# Interface class to the different lepton injectors

from typing import Any, Iterable, List, Union

import awkward as ak
import numpy as np

from .. import Particle
from ..injection_event.injection_event import InjectionEvent


def recursive_getattr(x: Any, attr: str) -> Any:
    """Get an attribute that is farther down an object hierarchy.

    Examples
    --------
    ``recursive_getattr(obj, "a.b")`` is equivalent to
    ``getattr(getattr(obj, "a"), "b")``.

    Parameters
    ----------
    x : Any
        Base object.
    attr : str
        Period-delimited string of attributes to grab.

    Returns
    -------
    Any
        Retrieved attribute value.
    """
    for a in attr.split("."):
        x = getattr(x, a)
    return x


def recursively_get_final_property(
    particles: Iterable[Particle], attr: str, idx: Union[None, int] = None
) -> np.ndarray:
    """A helper for getting the attributes from particles.

    Parameters
    ----------
    particles : Iterable[Particle]
        Iterable with particles from which to extract the same attribute.
    attr : str
        Period-delimited string of attributes to grab.
    idx : int or None, optional
        If the final attribute is an iterable and you only want the value
        from a specific index, specify it here. This is useful for, e.g., getting
        the x-position from a 3-vector.

    Returns
    -------
    l : np.ndarray
        A numpy array with the requested attribute for each particle. The shape
        of this array is equal to the length of the ``particles`` input parameter.
    """
    # Collect values in a Python list to avoid repeated NumPy concatenation
    acc = []

    def _collect(ps: Iterable[Particle]):
        for p in ps:
            try:
                val = recursive_getattr(p, attr)
            except Exception:
                # If attribute lookup fails for a particle, skip it
                continue
            if idx is not None:
                try:
                    val = val[idx]
                except Exception:
                    # If indexing fails, fall back to the raw value
                    pass
            # Ensure we append a 1-D numpy array for consistent concatenation
            arr = np.atleast_1d(np.asarray(val))
            acc.append(arr)
            # Recurse into children
            children = getattr(p, "children", None)
            if children:
                _collect(children)

    _collect(particles)
    if not acc:
        return np.array([])
    return np.concatenate(acc)


class Injection:
    """Base class for Prometheus injection."""

    def __init__(self, events: Iterable[InjectionEvent]):
        """Initialize the injection object.

        Parameters
        ----------
        events : Iterable[InjectionEvent]
            List or iterable of injection events.
        """
        self._events = events
        self._size = len(events)
        self._current_idx = 0

    def __getitem__(self, idx) -> InjectionEvent:
        return self.events[idx]

    def __len__(self) -> int:
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
    def events(self) -> List[InjectionEvent]:
        return self._events

    def to_dict(self) -> dict:
        """Convert all the properties of the injection to a dictionary."""
        d = {}
        d["interaction"] = [x.interaction.value for x in self]
        d["initial_state_energy"] = [x.initial_state.e for x in self]
        d["initial_state_type"] = [x.initial_state.pdg_code for x in self]
        d["initial_state_zenith"] = [x.initial_state.theta for x in self]
        d["initial_state_azimuth"] = [x.initial_state.phi for x in self]
        d["initial_state_x"] = [x.initial_state.position[0] for x in self]
        d["initial_state_y"] = [x.initial_state.position[1] for x in self]
        d["initial_state_z"] = [x.initial_state.position[2] for x in self]

        # Extract final-state properties in a single traversal per event to
        # avoid repeated recursive scans (previously called recursively_get_final_property
        # seven times per event).
        def _extract_final_state_props(particles):
            e_list = []
            pdg_list = []
            theta_list = []
            phi_list = []
            x_list = []
            y_list = []
            z_list = []
            parent_list = []

            def _collect(ps):
                for p in ps:
                    try:
                        e_list.append(p.e)
                    except Exception:
                        e_list.append(np.nan)
                    try:
                        pdg_list.append(p.pdg_code)
                    except Exception:
                        pdg_list.append(None)
                    try:
                        theta_list.append(p.theta)
                    except Exception:
                        theta_list.append(np.nan)
                    try:
                        phi_list.append(p.phi)
                    except Exception:
                        phi_list.append(np.nan)
                    try:
                        pos = p.position
                        x_list.append(pos[0])
                        y_list.append(pos[1])
                        z_list.append(pos[2])
                    except Exception:
                        x_list.append(np.nan)
                        y_list.append(np.nan)
                        z_list.append(np.nan)
                    try:
                        parent_list.append(getattr(p.parent, "serialization_idx", None))
                    except Exception:
                        parent_list.append(None)
                    children = getattr(p, "children", None)
                    if children:
                        _collect(children)

            _collect(particles)
            return {
                "e": np.array(e_list, dtype=float),
                "pdg": np.array(pdg_list, dtype=object),
                "theta": np.array(theta_list, dtype=float),
                "phi": np.array(phi_list, dtype=float),
                "x": np.array(x_list, dtype=float),
                "y": np.array(y_list, dtype=float),
                "z": np.array(z_list, dtype=float),
                "parent": np.array(parent_list, dtype=object),
            }

        final_state_es = []
        final_state_types = []
        final_state_zeniths = []
        final_state_azimuths = []
        final_state_xs = []
        final_state_ys = []
        final_state_zs = []
        parents = []
        for event in self:
            props = _extract_final_state_props(event.final_states)
            final_state_es.append(props["e"])
            final_state_types.append(props["pdg"])
            final_state_zeniths.append(props["theta"])
            final_state_azimuths.append(props["phi"])
            final_state_xs.append(props["x"])
            final_state_ys.append(props["y"])
            final_state_zs.append(props["z"])
            parents.append(props["parent"])
        d["final_state_energy"] = final_state_es
        d["final_state_type"] = final_state_types
        d["final_state_zenith"] = final_state_zeniths
        d["final_state_azimuth"] = final_state_azimuths
        d["final_state_x"] = final_state_xs
        d["final_state_y"] = final_state_ys
        d["final_state_z"] = final_state_zs
        d["final_state_parent"] = parents

        return d

    def to_awkward(self) -> ak.Array:
        """Convert all the properties of the injection to an Awkward array."""
        return ak.Array(self.to_dict())
