# -*- coding: utf-8 -*-
# detector_handler.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Deals with detector stuff

import sys
import numpy as np
from .config import config
sys.path.append('../')
from olympus.event_generation.detector import (  # noqa: E402
    Detector, Module
)
from scipy.stats import gamma

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

    def make_detector_from_file(
            self, random_state_seed=1337, noise_rate=1e-6,
            efficiency=0.2) -> None:
        """ Same as make_detector, just uses a plain csv file to load
        coordinates
        """
        tmp = np.loadtxt(config["detector"]['detector specs file'])
        xy_coords = np.column_stack((tmp[0], tmp[1]))
        self.make_detector(xy_coords, tmp[2].astype(int), tmp[3])

    def make_detector(
            self, xy_coords, nz_list, dist_z_list,
            random_state_seed=1337, noise_rate=1e-6,
            efficiency=0.2) -> None:
        """ make a simple detector

        Construct a detector made up of lines.

        Parameters
        ----------
        xy_coords: np.array
            The xy coordinates of the lines
        nz_list: np.array
            Number of modules per line
        dist_z_list: np.array
            The distances between the modules for each line.
            Equidistance is assumed per line
        random_state_seed: int
            The random state seed to use for the noise sampling
        noise_rate: float
            Baseline noise rate in 1/ns.
            Will be multiplied to gamma(1, 0.25) distributed 
            random rates per module.

        Returns
        -------
        None

        Notes
        -----
        This will call self.to_f2k and create a detector file for future use
        """
        modules = []

        for id_line, xy in enumerate(xy_coords):
            modules += self.make_line(
                xy[0], xy[1], nz_list[id_line],
                dist_z_list[id_line],
                id_line,
                np.random.RandomState(random_state_seed),
                baseline_noise_rate=noise_rate,
                efficiency=efficiency
            )
        d = Detector(modules)
        self.to_f2k(d, self._fname)

    def make_line(
            self, x, y, n_z, dist_z, line_id,
            rng,
            baseline_noise_rate=1e-6,
            efficiency=0.2):
        """
        Make a line of detector modules.

        The modules share the same (x, y) coordinate and are spaced along the z-direction.

        Parameters:
            x, y: float
                (x, y) position of the line
            n_z: int
                Number of modules per line
            dist_z: float
                Spacing of the detector modules in z
            rng: RandomState
            baseline_noise_rate: float
                Baseline noise rate in 1/ns. Will be multiplied to gamma(1, 0.25) distributed
                random rates per module.
            line_id: int
                Identifier for this line
        """
        modules = []
        for i, pos_z in enumerate(np.linspace(-dist_z * n_z / 2, dist_z * n_z / 2, n_z)):
            pos = np.array([x, y, pos_z])
            noise_rate = (
                gamma.rvs(1, 0.25, random_state=rng) * baseline_noise_rate
            )
            mod = Module(
                pos, key=(line_id, i), noise_rate=noise_rate, efficiency=efficiency
            )
            modules.append(mod)
        return modules
