# -*- coding: utf-8 -*-
"""
Definition file generator.

Generates documentation files containing function definitions and parameter
conversions for the fennel parametrization. Helps with inspecting and
exporting the parametrization data.
"""

import collections
import inspect

# Imports
import logging
import pickle
import pkgutil
from typing import Dict

import pandas as pd

# Local imports
from .config import config
from .em_cascades import EM_Cascade
from .hadron_cascades import Hadron_Cascade
from .tracks import Track

_log = logging.getLogger(__name__)


class Definitions_Generator:
    """
    Generate definition files from parametrization data.

    Creates human-readable documentation of all parametrization functions
    and can export parameter files to CSV format.

    Attributes
    ----------
    _fname : str
        Output filename for definitions
    _lines_to_write : list
        Lines to write to definition file

    Examples
    --------
    >>> from fennel import Fennel
    >>> f = Fennel()
    >>> # Definition generator used internally

    Notes
    -----
    Useful for inspecting parametrization structure and exporting data.
    """

    def __init__(
        self, track: Track, em_cascade: EM_Cascade, hadron_cascade: Hadron_Cascade
    ) -> None:
        """
        Initialize the definitions generator.

        Parameters
        ----------
        track : Track
            Track calculator instance
        em_cascade : EM_Cascade
            EM cascade calculator instance
        hadron_cascade : Hadron_Cascade
            Hadron cascade calculator instance
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        self._fname = config["advanced"]["generated definitions"]
        self._lines_to_write = []
        # The tracks
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        self._lines_to_write.append(
            "# Tracks\n",
        )
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        for val in track.__dict__.values():
            if callable(val):
                self._lines_to_write.append(inspect.getsource(val))
        # The em cascades
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        self._lines_to_write.append(
            "# EM Cascades\n",
        )
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        for val in em_cascade.__dict__.values():
            if callable(val):
                self._lines_to_write.append(inspect.getsource(val))
        # The hadron cascades
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        self._lines_to_write.append(
            "# Hadronic Cascades\n",
        )
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        for val in hadron_cascade.__dict__.values():
            if callable(val):
                self._lines_to_write.append(inspect.getsource(val))

    def _write(self):
        """Write the definitions file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with open(self._fname, "w") as f:
            for line in self._lines_to_write:
                f.write(line)

    def _pars2csv(self):
        """Converts the calculation parameters to a csv file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        param_file = pkgutil.get_data(
            __name__, "data/%s.pkl" % config["scenario"]["parametrization"]
        )
        params = pickle.loads(param_file)
        # Flatten
        params = self._flatten(params)
        pd.DataFrame.from_dict(data=params, orient="index").to_csv(
            "parameters.csv", header=False
        )

    def _flatten(self, d: Dict, parent_key="", sep="_"):
        """Helper function to flatten a dictionary of dictionaries

        Parameters
        ----------
        d : Dict
            The dictionary to flatten
        parent_key : str
            Optional: Key in the parent dictionary
        sep : str
            The seperator used

        Returns
        -------
        flattened_dic : dic
            The flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + str(k) if parent_key else str(k)
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(self._flatten(v, str(new_key), sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
