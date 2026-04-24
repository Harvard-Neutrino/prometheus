"""Small pickle loader that maps Haiku pickled types to plain Python types.

This Unpickler maps the Haiku `FlatMapping` wrapper used in the saved
parameter pickles to a plain mapping (dict) so tests can `load()` the
resources without importing `dm-haiku`.

Usage:
    from prometheus.compat.haiku_unpickler import load as haiku_load
    config, params = haiku_load(path_to_pickle)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import BinaryIO, Any, Union


class _HaikuUnpickler(pickle.Unpickler):
    """Pickle unpickler that transparently maps Haiku ``FlatMapping`` to plain dicts."""

    def find_class(self, module: str, name: str):
        """Return a constructor for ``module.name``, remapping Haiku types to plain mappings.

        Parameters
        ----------
        module : str
            Module name as stored in the pickle stream.
        name : str
            Class name as stored in the pickle stream.

        Returns
        -------
        constructor : callable
            Constructor for the requested type, or an identity function for
            Haiku ``FlatMapping``/``FlatMap`` wrappers.
        """
        # Map the Haiku FlatMapping/FlatMap wrapper used in older pickles
        # to a simple constructor that returns the underlying mapping.
        if module in ("haiku._src.data_structures", "haiku.data_structures"):
            if name in ("FlatMapping", "FlatMap"):
                return lambda mapping: mapping
        return super().find_class(module, name)


def load(f: Union[str, Path, BinaryIO]) -> Any:
    """Load a pickle while transparently handling Haiku FlatMapping.

    Parameters
    ----------
    f : str, Path, or BinaryIO
        File path or binary file-like object.

    Returns
    -------
    result : Any
        The unpickled Python object, with Haiku ``FlatMapping`` converted to
        plain mappings (dict-like).
    """
    if isinstance(f, (str, Path)):
        with open(f, "rb") as fh:
            return _HaikuUnpickler(fh).load()
    return _HaikuUnpickler(f).load()
