"""Photon-propagator plugin registry.

To add a new photon propagator:
1. Decorate the class with ``@register_propagator("my_name")``.
2. That's it — no other files need to change.
"""
from __future__ import annotations

from typing import Type

_PROPAGATOR_REGISTRY: dict[str, Type] = {}


def register_propagator(name: str):
    """Class decorator that registers a photon propagator under *name*."""
    def decorator(cls):
        _PROPAGATOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_propagator(name: str) -> Type:
    """Return the propagator class for *name*, loading it lazily on first call.

    Parameters
    ----------
    name : str
        Canonical propagator name (case-insensitive), e.g. ``"olympus"``,
        ``"ppc"``, or ``"ppc_cuda"``.

    Raises
    ------
    UnknownPhotonPropagatorError
        If *name* is not registered and cannot be resolved by lazy import.
    """
    from ..utils import UnknownPhotonPropagatorError

    name_lower = name.lower()
    if name_lower not in _PROPAGATOR_REGISTRY:
        if name_lower == "olympus":
            from .olympus_photon_propagator import OlympusPhotonPropagator  # noqa: F401 – side-effect: registers class
        elif name_lower in ("ppc", "ppc_cuda"):
            from .ppc_photon_propagator import PPCPhotonPropagator  # noqa: F401
        else:
            raise UnknownPhotonPropagatorError(f"Unknown photon propagator: {name!r}")
    try:
        return _PROPAGATOR_REGISTRY[name_lower]
    except KeyError:
        raise UnknownPhotonPropagatorError(f"Unknown photon propagator: {name!r}")
