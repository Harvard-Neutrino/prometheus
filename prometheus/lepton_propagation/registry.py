"""Lepton propagator plugin registry.

To add a new lepton propagator:
1. Decorate the class with ``@register_lepton_propagator("my_name")``.
2. That's it — no other files need to change.
"""

from __future__ import annotations

from typing import Type

_LEPTON_PROPAGATOR_REGISTRY: dict[str, Type] = {}


def register_lepton_propagator(name: str):
    """Class decorator that registers a lepton propagator under ``name``.

    The name is normalised to lower-case with spaces replaced by underscores.
    """

    def decorator(cls):
        _LEPTON_PROPAGATOR_REGISTRY[name.lower().replace(" ", "_")] = cls
        return cls

    return decorator


def get_lepton_propagator(name: str) -> Type:
    """Return the lepton propagator class for ``name``, loading it lazily.

    Parameters
    ----------
    name : str
        Canonical propagator name, e.g. "new proposal" or
        "new_proposal".

    Raises
    ------
    ValueError
        If ``name`` is not registered and cannot be resolved by lazy import.
    """
    name_key = name.lower().replace(" ", "_")
    if name_key not in _LEPTON_PROPAGATOR_REGISTRY:
        if "proposal" in name_key:
            from .new_proposal_lepton_propagator import NewProposalLeptonPropagator  # noqa: F401
        else:
            raise ValueError(f"Unknown lepton propagator: {name!r}")
    try:
        return _LEPTON_PROPAGATOR_REGISTRY[name_key]
    except KeyError:
        raise ValueError(f"Unknown lepton propagator: {name!r}")
