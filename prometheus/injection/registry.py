"""Injector plugin registry.

To add a new injector:
1. Decorate the injector *runner* function (or class) with
   ``@register_injector("my_name", constructor=<from-file callable>)``.
2. That's it — no other files need to change.

This registry is intentionally minimal.
"""

from __future__ import annotations

from typing import Callable

_INJECTOR_REGISTRY: dict[str, Callable] = {}
_INJECTION_CONSTRUCTOR_REGISTRY: dict[str, Callable] = {}


def register_injector(name: str, *, constructor: Callable | None = None):
    """Decorator that registers an injector runner under *name*.

    Parameters
    ----------
    name : str
        Canonical injector name (case-insensitive).
    constructor : callable, optional
        Function that reads an injection from a file produced by this
        injector (e.g. ``injection_from_LI_output``).
    """

    def decorator(fn):
        _INJECTOR_REGISTRY[name.lower()] = fn
        if constructor is not None:
            _INJECTION_CONSTRUCTOR_REGISTRY[name.lower()] = constructor
        return fn

    return decorator


def get_injector(name: str) -> Callable:
    """Return the injector runner for *name*.

    Raises
    ------
    ValueError
        If *name* is not a registered injector.
    """
    key = name.lower()
    if key not in _INJECTOR_REGISTRY:
        if key == "leptoninjector":
            from .lepton_injector_utils import make_new_LI_injection  # noqa: F401
        else:
            raise ValueError(f"Unknown injector: {name!r}")
    try:
        return _INJECTOR_REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown injector: {name!r}")
