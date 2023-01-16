# -*- coding: utf-8 -*-
# detector_handler.py
# Copyright (C) 2022 Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Deals with detector stuff

import numpy as np
import awkward as ak
from typing import List, Union

from .module import Module
from .medium import Medium
from ..config import config

class Detector(object):
    """Interface for detector methods
    """
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

    def __getitem__(self, key):
        idx = self._om_keys.index(key)
        return self.modules[idx]

    def __add__(self, other):
        modules = np.hcat(self.modules, other.modules)
        return Detector(modules)

    @property
    def medium(self):
        return self._medium

    @property
    def modules(self):
        return self._modules

    @property
    def n_modules(self):
        return self._n_modules

    @property
    def outer_radius(self):
        return self._outer_radius

    @property
    def outer_cylinder(self):
        return self._outer_cylinder

    @property
    def offset(self):
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
            format, but there exact value does not matter
        mac_ids: MAC (I don't think this is actually what this is called) IDs for the DOMs
        """
        if serial_nos and len(serial_nos)!=len(self.modules):
            raise ValueError("serial numbers incompatible with modules")
        if mac_ids and len(mac_ids)!=len(self.modules):
            raise ValueError("mac id list incompatible with modules")
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