# -*- coding: utf-8 -*-
# detector_handler.py
# Copyright (C) 2022 Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Deals with detector stuff
from __future__ import annotations

import warnings
from typing import List, Tuple, Union

import awkward as ak
import numpy as np

from .medium import Medium
from .module import Module


class IncompatibleSerialNumbersError(Exception):
    """Raised when serial numbers length doesn't match number of DOMs."""

    def __init__(self):
        self.message = "Serial numbers incompatible with modules"
        super().__init__(self.message)


class IncompatibleMACIDsError(Exception):
    """Raised when MAC IDs length doesn't match number of DOMs."""

    def __init__(self):
        self.message = "MAC IDs incompatible with modules"
        super().__init__(self.message)


class GeometryError(ValueError):
    """Raised when a detector geometry is not usable.

    The message lists every problem found so a geo file can be fixed in one
    pass rather than one error at a time.
    """

    def __init__(self, problems: List[str], source: str = "detector geometry"):
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"Invalid {source}:\n{lines}")


# Modules on one string may drift in (x, y) by well under a metre in surveyed
# geometries (IceCube: 0.6 m). Anything past this is a different string.
STRING_XY_TOLERANCE_M = 5.0


def _format_examples(items: List[str], limit: int = 5) -> str:
    """Join up to ``limit`` example strings, noting how many were left out."""
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        shown += f", ... ({len(items) - limit} more)"
    return shown


def validate_modules(modules: List[Module]) -> None:
    """Check that a list of modules describes a usable detector.

    Errors are collected and raised together. Every ``(string_id, om_id)`` key
    must be unique because hits are matched back to modules by key, and every
    position must be finite. Negative IDs and strings whose modules spread
    more than ``STRING_XY_TOLERANCE_M`` in (x, y) are legal but unusual, so they only warn.

    Parameters
    ----------
    modules : list of Module
        Modules to check.

    Raises
    ------
    GeometryError
        If the modules cannot form a valid detector.
    """
    problems = []
    if len(modules) == 0:
        raise GeometryError(["no modules"])

    pos = np.array([m.pos for m in modules], dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        problems.append(f"module positions must be (x, y, z); got shape {pos.shape}")
    else:
        bad = np.where(~np.isfinite(pos).all(axis=1))[0]
        if len(bad):
            problems.append(
                f"{len(bad)} module(s) with non-finite position, e.g. module index "
                f"{_format_examples([str(i) for i in bad])}"
            )

    # Duplicate keys. Explain them by the physical string they came from: the
    # usual cause is one string ID reused for two strings at different (x, y).
    seen = {}
    dup_keys = []
    for m in modules:
        if m.key in seen:
            dup_keys.append(m.key)
        else:
            seen[m.key] = m
    if dup_keys:
        problems.append(
            f"{len(dup_keys)} duplicate (string_id, om_id) key(s), e.g. "
            f"{_format_examples([str(k) for k in dup_keys])}"
        )
    strings = {}
    for m in modules:
        strings.setdefault(m.key[0], []).append(m)
    reused = []
    unaligned = []
    for sid, mods in strings.items():
        xy_all = np.array([m.pos[:2] for m in mods], dtype=float)
        if np.ptp(xy_all, axis=0).max() > STRING_XY_TOLERANCE_M:
            xy = np.unique(np.round(xy_all / STRING_XY_TOLERANCE_M) * STRING_XY_TOLERANCE_M, axis=0)
            where = "; ".join(f"({x:g}, {y:g})" for x, y in xy[:3])
            keys = {m.key for m in mods}
            if len(keys) < len(mods):
                reused.append(f"string {sid} at {where}")
            else:
                unaligned.append(f"string {sid} at {where}")
    if reused:
        problems.append(
            f"{len(reused)} string ID(s) shared by modules at different (x, y), so two "
            f"physical strings carry the same ID: {_format_examples(reused, 3)}"
        )

    if problems:
        raise GeometryError(problems)

    negative = sorted({m.key for m in modules if m.key[0] < 0 or m.key[1] < 0})
    if negative:
        warnings.warn(
            f"{len(negative)} module(s) with a negative string or OM ID, e.g. "
            f"{_format_examples([str(k) for k in negative])}. Prometheus accepts these "
            "but downstream tools may not.",
            stacklevel=3,
        )
    if unaligned:
        warnings.warn(
            f"{len(unaligned)} string ID(s) whose modules are not on one vertical line: "
            f"{_format_examples(unaligned, 3)}",
            stacklevel=3,
        )


class Detector(object):
    """Prometheus detector object."""

    def __init__(self, modules: List[Module], medium: Union[Medium, None]):
        """Initialize detector.

        Parameters
        ----------
        modules : list of Module
            List of all the modules in the detector.
        medium : Medium or None
            Medium in which the detector is embedded.
        """
        validate_modules(modules)
        self._modules = modules
        self._medium = medium
        self._offset = np.mean(np.array([m.pos for m in modules]), axis=0)
        self.module_coords = np.vstack([m.pos for m in self.modules])
        self.module_coords_ak = ak.Array(self.module_coords)
        self.module_efficiencies = np.asarray([m.efficiency for m in self.modules])
        self.module_noise_rates = np.asarray([m.noise_rate for m in self.modules])

        # TODO replace this with the functions David writes
        self._outer_radius = np.linalg.norm(self.module_coords - self.offset, axis=1).max()
        self._outer_cylinder = (
            np.linalg.norm(self.module_coords[:, :2] - self.offset[:2].transpose(), axis=1).max(),
            self.module_coords[:, 2].max() - self.module_coords[:, 2].min(),
        )
        self._n_modules = len(modules)
        self._om_keys = [om.key for om in self.modules]

    def __getitem__(self, key) -> Module:
        idx = self._om_keys.index(key)
        return self.modules[idx]

    def __add__(self, other) -> Detector:
        if self.medium != other.medium:
            raise ValueError("Cannot combine detectors that are in different media")
        modules = self.modules + other.modules
        return Detector(modules, self.medium)

    @property
    def medium(self) -> Medium:
        return self._medium

    @property
    def modules(self) -> List[Module]:
        return self._modules

    @property
    def n_modules(self) -> int:
        return self._n_modules

    @property
    def outer_radius(self) -> float:
        return self._outer_radius

    @property
    def outer_cylinder(self) -> Tuple[float, float]:
        return self._outer_cylinder

    @property
    def offset(self) -> np.ndarray:
        return self._offset

    def to_f2k(self, geo_file: str, serial_nos: List[str] = [], mac_ids: List[str] = []) -> None:
        """Write detector coordinates into f2k format.

        Parameters
        ----------
        geo_file : str
            Filepath of the output geometry file.
        serial_nos : list of str, optional
            Serial numbers for the optical modules. These MUST be in
            hexadecimal format, but their exact value does not matter. If
            nothing is provided, these values will be randomly generated.
        mac_ids : list of str, optional
            MAC (I don't think this is actually what this is called) IDs
            for the DOMs. By default these will be randomly generated, which
            is probably what you want to do.

        Raises
        ------
        IncompatibleSerialNumbersError
            Raised if serial numbers length doesn't match number of DOMs.
        IncompatibleMACIDsError
            Raised if MAC IDs length doesn't match number of DOMs.
        """
        if serial_nos and len(serial_nos) != len(self.modules):
            raise IncompatibleSerialNumbersError()

        if mac_ids and len(mac_ids) != len(self.modules):
            raise IncompatibleMACIDsError()

        # Make serial numbers place holders
        if not serial_nos:
            from .utils import random_serial

            serial_nos = [random_serial() for _ in range(self.n_modules)]

        # Make MAC ID place holders
        if not mac_ids:
            from .utils import random_mac

            mac_ids = [random_mac() for _ in range(self.n_modules)]

        keys = [m.key for m in self.modules]
        iterable = zip(mac_ids, serial_nos, self.module_coords, keys)
        with open(geo_file, "w") as f2k_out:
            for mac_id, serial_no, pos, key in iterable:
                line = f"{mac_id}\t{serial_no}\t{pos[0]}\t{pos[1]}\t{pos[2]}"
                if hasattr(key, "__iter__"):
                    for x in key:
                        line += f"\t{x}"
                else:
                    line += f"\t{key}"
                line += "\n"
                f2k_out.write(line)

    def display(self, ax=None, elevation_angle=0, azimuth=0):
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        ax.scatter(
            self.module_coords[:, 0],
            self.module_coords[:, 1],
            self.module_coords[:, 2],
            alpha=0.5,
            s=0.2,
        )
        ax.view_init(np.degrees(elevation_angle), np.degrees(azimuth))
        plt.show()

    def to_geo(self, geofile):
        with open(geofile, "w") as f:
            f.write("### Metadata ###\n")
            f.write(f"Medium:\t{self.medium.name.lower()}\n")
            f.write("### Modules ###\n")
            for module in self.modules:
                line = f"{module.pos[0]}\t{module.pos[1]}\t{module.pos[2]}"
                for x in module.key:
                    line += f"\t{x}"
                line += "\n"
                f.write(line)
