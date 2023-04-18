# -*- coding: utf-8 -*-
# detector_handler.py
# Copyright (C) 2022 Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Deals with detector stuff
from __future__ import annotations

import numpy as np
import awkward as ak
from typing import List, Union, Tuple

from .module import Module
from .medium import Medium
from ..config import config

class IncompatibleSerialNumbersError(Exception):
    """Raised when serial numbers length doesn't match number of DOMs"""
    def __init__(self):
        self.message = "Serial numbers incompatible with modules"
        super().__init__(self.message)

class IncompatibleMACIDsError(Exception):
    """Raised when MAC IDs length doesn't match number of DOMs"""
    def __init__(self):
        self.message = "MAC IDs incompatible with modules"
        super().__init__(self.message)

class Detector(object):
    """Prometheus detector object"""
    def __init__(self, modules: List[Module], medium: Union[Medium, None]):
        """Initialize detector.
        params
        ______
        modules: List of all the modules in the detector
        medium: Medium in which the detector is embedded
        """
        self._modules = modules
        self._medium = medium
        self._offset = np.mean(np.array([m.pos for m in modules]), axis=0)
        self.module_coords = np.vstack([m.pos for m in self.modules])
        self.module_coords_ak = ak.Array(self.module_coords)
        self.module_efficiencies = np.asarray([m.efficiency for m in self.modules])
        self.module_noise_rates = np.asarray([m.noise_rate for m in self.modules])
        
        # TODO replace this with the functions David writes
        self._outer_radius = np.linalg.norm(self.module_coords-self.offset, axis=1).max()
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
        if self.medium!=other.medium:
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

    def to_f2k(
        self,
        geo_file: str,
        serial_nos: List[str]=[],
        mac_ids: List[str]=[]
    ) -> None:
        """Write detector corrdinates into f2k format.
        
        params
        ______
        geo_file: file name of where to write it
        serial_nos: serial numbers for the optical modules. These MUST be in hexadecimal
            format, but there exact value does not matter. If nothing is provided, these
            values will be randomly generated
        mac_ids: MAC (I don't think this is actually what this is called) IDs for the DOMs.
            By default these will be randomly generated. This is prbably what you want
            to do.

        raises
        ______

        """
        if serial_nos and len(serial_nos)!=len(self.modules):
            raise IncompatibleSerialNumbersError()

        if mac_ids and len(mac_ids)!=len(self.modules):
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
            ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.scatter(
            self.module_coords[:,0],
            self.module_coords[:,1],
            self.module_coords[:,2],
            alpha=0.5,
            s=0.2
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

