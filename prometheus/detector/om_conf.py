"""Read PPC ``om.conf`` files into optical-module definitions."""

from pathlib import Path
from typing import Dict, Union


def read_om_conf(path: Union[str, Path]) -> Dict[int, dict]:
    """Read module definitions from a PPC ``om.conf`` file.

    Parameters
    ----------
    path : str or Path
        File to read.

    Returns
    -------
    dict
        Maps each type ID to the keyword arguments that make a ``Module`` of that
        type. Every entry has ``module_type``, ``area``, ``beta``, ``Rr``, ``Rz``,
        ``n_pmts``, ``pmt_dirs`` and ``cable_azimuth``. As in PPC, the first
        definition of a repeated type ID is kept.

    Raises
    ------
    ValueError
        If a definition is malformed or has too few PMT directions.
    """
    with open(path) as f:
        lines = [line.split("#", 1)[0].split() for line in f]
    lines = [tokens for tokens in lines if tokens]

    types: Dict[int, dict] = {}
    i = 0
    while i < len(lines):
        tokens = lines[i]
        i += 1
        if len(tokens) < 9:
            raise ValueError(f"Malformed module definition in {path}: {' '.join(tokens)!r}")
        n_pmts = int(tokens[6])
        pmt_dirs = [(float(tokens[7]), float(tokens[8]))]
        for extra in lines[i : i + n_pmts - 1]:
            # A module header starts with a name, so it has more than two columns.
            if len(extra) != 2:
                break
            pmt_dirs.append((float(extra[0]), float(extra[1])))
        if len(pmt_dirs) != n_pmts:
            raise ValueError(
                f"Module {tokens[0]} in {path} declares {n_pmts} PMTs but lists {len(pmt_dirs)}"
            )
        i += n_pmts - 1
        types.setdefault(
            int(tokens[1]),
            {
                "module_type": int(tokens[1]),
                "area": float(tokens[2]),
                "beta": float(tokens[3]),
                "Rr": float(tokens[4]),
                "Rz": float(tokens[5]),
                "n_pmts": n_pmts,
                "pmt_dirs": pmt_dirs,
                "cable_azimuth": float(tokens[9]) if len(tokens) > 9 else None,
            },
        )
    return types
