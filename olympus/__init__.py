"""Backward-compatibility shim.

The olympus implementation now lives in
``prometheus.photon_propagation.olympus``.  This shim pre-populates
``sys.modules`` so that existing code using ``import olympus.X`` continues to
work without modification.
"""
import importlib
import sys

# Only pre-register submodules that are imported by external code.
# Heavy or optional modules (plotting, optimization) are left out to avoid
# importing unavailable optional dependencies (e.g. plotly).
_submodules = [
    "event_generation",
    "event_generation.mc_record",
    "event_generation.photon_source",
]

for _sub in _submodules:
    _key = f"olympus.{_sub}"
    if _key not in sys.modules:
        sys.modules[_key] = importlib.import_module(
            f"prometheus.photon_propagation.olympus.{_sub}"
        )
