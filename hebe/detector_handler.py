# -*- coding: utf-8 -*-
# detector_handler.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Deals with detector stuff

import sys
import numpy as np
from config import config
sys.path.append('../')
from olympus.event_generation.detector import (  # noqa: E402
    Detector, Module
)

class DH(object):
    """ Interface for detector methods
    """
    def __init__(self):
        print('Starting up the detector handler')
        self._fname = config['detector']['file name']

    def to_f2k(
            self,
            d:Detector,
            geo_file,
            serial_nos=[], mac_ids=[]) -> None:
        """
        Write detector corrdinates into f2k format.
        Parameters
            d: Detector
                Detector object to write out
            serial_nos: list{str}
                serial numbers for the optical modules
            mac_ids: list{str}
                MAC (I don't think this is actually
                what this is called) IDs for the DOMs
        """
        # Make sure serial numbers are compatible with the list of OMs
        if serial_nos:
            if len(serial_nos)!=len(d.modules):
                raise ValueError("serial numbers incompatible with modules")
        # Make up random place holders
        else:
            import string, random
            serial_nos = ["0x"+"".join(random.choices(
                string.ascii_lowercase + string.digits, k=12
            )) 
                    for _ in range(len(d.modules))
                    ]
        # Make sure serial numbers are compatible with the list of OMs
        if mac_ids:
            if len(mac_ids)!=len(d.modules):
                raise ValueError("mac id list incompatible with modules")
        # Make up random place holders
        else:
            import string, random
            mac_ids = [''.join(random.choices(
                string.ascii_uppercase + string.digits, k=8)) 
                    for _ in range(len(d.modules))
                    ]
            keys = [m.key for m in d.modules]
        with open(geo_file, "w") as f2k_out:
            for mac_id, serial_no, pos, key in zip(
                    mac_ids, serial_nos, d.module_coords, keys):
                line = f"{mac_id}\t{serial_no}\t{pos[0]}\t{pos[1]}\t{pos[2]}"
                if hasattr(key, "__iter__"):
                    for x in key:
                        line += f"\t{x}"
                else:
                    line += f"\t{key}"
                line += "\n"
                f2k_out.write(line)

    def from_f2k(
            self,
            efficiency=0.2,
            noise_rate=1) -> Detector:
        """
        Create a Detector object from an f2k geometry file
        Parameters
            self.key: collection
                Module identifier
            noise_rate: float
                Noise rate in 1/ns
            efficiency: float
                Module efficiency (0, 1]
        Returns
            d: Detector
        """
        # List for the (x,y,z) positions of each OM
        pos = []
        # List for the optical module keys
        keys = []
        # List for the serial numbers
        sers = []
        with open(self._fname) as f2k_in:
            for line in f2k_in.readlines():
                line = line.strip("\n").split("\t")
                sers.append(line[1])
                pos.append(
                    np.array([float(line[2]), float(line[3]),
                    float(line[4])]))
                keys.append((int(line[5]), int(line[6])))
        if not hasattr(efficiency, "__iter__"):
            efficiency = np.full(len(pos), efficiency)
        if not hasattr(noise_rate, "__iter__"):
            noise_rate = np.full(len(pos), noise_rate)
        modules = [
            Module(p, k, efficiency=e, noise_rate=nr, serial_no=ser) 
                for p,k,e,nr,ser in zip(pos, keys, efficiency, noise_rate, sers
            )]
        d = Detector(modules)
        return d
