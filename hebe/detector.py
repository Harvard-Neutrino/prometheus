# -*- coding: utf-8 -*-
# detector_handler.py
# Copyright (C) 2022 Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Deals with detector stuff

import numpy as np
import awkward as ak
from .config import config
from .utils import iter_or_rep


class Module(object):
    """
    Detection module.
    Attributes:
        pos: np.ndarray
            Module position (x, y, z)
        noise_rate: float
            Noise rate in 1/ns
        efficiency: float
            Module efficiency (0, 1]
        self.key: collection
            Module identifier
    """

    def __init__(self, pos, key, noise_rate=1, efficiency=0.2, serial_no=None):
        """Initialize a module."""
        self.pos = pos
        self.noise_rate = noise_rate
        self.efficiency = efficiency
        self.key = key
        self.serial_no = serial_no

    def __repr__(self):
        """Return string representation."""
        return repr(
            f"Module {self.key}, {self.pos} [m], {self.noise_rate} [Hz], {self.efficiency}"
        )

class Detector(object):
    """ Interface for detector methods
    """
    def __init__(self, modules):
        """Initialize detector."""
        #self.modules = modules
        #self.module_coords = np.vstack([m.pos for m in self.modules])
        # We need to move all our modules to a coordinate system
        # Where (0,0,0) is the center of the detector
        self._offset = np.mean(np.array([m.pos for m in modules]), axis=0)
        self.modules = [Module(m.pos-self._offset, m.key,
            noise_rate=m.noise_rate, efficiency=m.efficiency, serial_no=m.serial_no) 
        for m in modules]
        self.module_coords = np.vstack([m.pos for m in self.modules])
        self.module_coords_ak = ak.Array(self.module_coords)
        self.module_efficiencies = np.asarray([m.efficiency for m in self.modules])
        self.module_noise_rates = np.asarray([m.noise_rate for m in self.modules])
        
        # TODO replace this with the functions David writes
        self._outer_radius = np.linalg.norm(self.module_coords, axis=1).max()
        self._outer_cylinder = (
            np.linalg.norm(self.module_coords[:, :2], axis=1).max(),
            2 * np.abs(self.module_coords[:, 2].max()),
        )
        self._n_modules = len(modules)
        self._om_keys = [om.key for om in self.modules]
        print("============================")
        print(self.outer_radius)
        print("============================")

    def __getitem__(self, key):
        idx = self._om_keys.index(key)
        return self.modules[idx]

    def __add__(self, other):
        modules = np.hcat(self.modules, other.modules)
        return Detector(modules)

    #def subdetectors(self, nmodules):
    #    start = 0
    #    end = nmodules
    #    slc = slice(start, end)
    #    subdetectors = [Detector(self.modules[slc])]
    #    while end <= len(self.modules):
    #        start += nmodules
    #        end += nmodules
    #        slc = slice(start, end)
    #        print(slc)
    #        subdet = Detector(self.modules[slc])
    #        subdetectors.append(subdet)
    #    return subdetectors

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
            if len(serial_nos)!=len(self.modules):
                raise ValueError("serial numbers incompatible with modules")
        # Make up random place holders
        else:
            import string, random
            serial_nos = ["0x"+"".join(random.choices(
                '0123456789abcdef', k=12
            )) 
                    for _ in range(len(self.modules))
                    ]
        # Make sure serial numbers are compatible with the list of OMs
        if mac_ids:
            if len(mac_ids)!=len(self.modules):
                raise ValueError("mac id list incompatible with modules")
        # Make up random place holders
        else:
            import string, random
            mac_ids = [''.join(random.choices(
                string.ascii_uppercase + string.digits, k=8)) 
                    for _ in range(len(self.modules))
                    ]
            keys = [m.key for m in self.modules]
        with open(geo_file, "w") as f2k_out:
            for mac_id, serial_no, pos, key in zip(
                mac_ids, serial_nos, self.module_coords+self._offset, keys
            ):
                line = f"{mac_id}\t{serial_no}\t{pos[0]}\t{pos[1]}\t{pos[2]}"
                if hasattr(key, "__iter__"):
                    for x in key:
                        line += f"\t{x}"
                else:
                    line += f"\t{key}"
                line += "\n"
                f2k_out.write(line)

def detector_from_f2k(
    fname,
    efficiency=0.2,
    noise_rate=1
) -> Detector:
    """
    Create a Detector object from an f2k geometry file
    Parameters
    __________
        fname: str
            f2k-file where detector information is stored
        noise_rate: float
            Noise rate in 1/ns
        efficiency: float
            Module efficiency (0, 1]
    Returns
    _______
        det: Detector
    """
    # List for the (x,y,z) positions of each OM
    pos = []
    # List for the optical module keys
    keys = []
    # List for the serial numbers
    sers = []
    with open(fname) as f2k_in:
        for line in f2k_in.readlines():
            line = line.strip("\n").split("\t")
            sers.append(line[1])
            pos.append(
                np.array([float(line[2]), float(line[3]),
                float(line[4])]))
            keys.append((int(line[5]), int(line[6])))
    efficiency, noise_rate = iter_or_rep(efficiency), iter_or_rep(noise_rate)
    modules = [
        Module(p, k, efficiency=e, noise_rate=nr, serial_no=ser) 
            for p,k,e,nr,ser in zip(pos, keys, efficiency, noise_rate, sers
        )]
    det = Detector(modules)
    return det

def detector_from_geo(
    fname,
    efficiency=0.2,
    noise_rate=1
) -> Detector:
    """
    Create a Detector object from a geometry file
    Parameters
    __________
        fname: str
            geo-file where detector information is stored
        noise_rate: float
            Noise rate in 1/ns
        efficiency: float
            Module efficiency (0, 1]
    Returns
    _______
        det: Detector
    """
    pos = []; keys = []
    with open(fname) as geo_in:
        read_lines = geo_in.readlines()
        modules_i = read_lines.index("### Modules ###\n")   

        for line in read_lines[modules_i+1:]:
            line = line.strip("\n").split("\t")
            pos.append(
                np.array([float(line[0]), float(line[1]),
                float(line[2])]))
            pos_out = np.array(pos)
            keys.append((int(line[3]),int(line[4])))

    import string, random 
    random.seed(config["general"]["random state seed"])
    sers = ["0x"+"".join(random.choices(
        string.ascii_lowercase + string.digits, k=12,
    )) 
            for _ in range(len(pos))
            ]
    efficiency, noise_rate = iter_or_rep(efficiency), iter_or_rep(noise_rate)
    modules = [
        Module(p, k, efficiency=e, noise_rate=nr, serial_no=ser) 
            for p,k,e,nr,ser in zip(pos, keys, efficiency, noise_rate, sers
        )]
    det = Detector(modules)
    return det

def make_line(x, y, n_z, dist_z, rng, baseline_noise_rate, line_id, efficiency=0.2):
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
            scipy.stats.gamma.rvs(1, 0.25, random_state=rng) * baseline_noise_rate
        )
        mod = Module(
            pos, key=(line_id, i), noise_rate=noise_rate, efficiency=efficiency
        )
        modules.append(mod)
    return modules


def make_grid(
    n_side, dist, n_z, dist_z, baseline_noise_rate=1e-6, rng=np.random.RandomState(1337)
):
    """
    Build a square detector grid.
    Strings of detector modules are placed on a square grid.
    The noise rate for each module is randomöy sampled from a gamma distribution
    Paramaters:
      n_side
        Number of detector strings per side
      dist
        Spacing between strings [m]
      n_z
        Number of detector modules per string
      dist_z
        Distance of modules on a string [m]
      baseline_noise_rate
        Baseline noise rate (default 1E-6Hz)
    """
    modules = []
    x_pos = np.linspace(-n_side / 2 * dist, n_side / 2 * dist, n_side)
    y_pos = x_pos

    for x, y in itertools.product(x_pos, y_pos):
        modules += make_line(x, y, n_z, dist_z, rng, baseline_noise_rate)

    return modules


def make_hex_grid(
    n_side, dist, n_z, dist_z, baseline_noise_rate=1e-6, rng=np.random.RandomState(1337)
):
    """
    Build a hex detector grid.
    Strings of detector modules are placed on a square grid.
    The noise rate for each module is randomöy sampled from a gamma distribution
    Paramaters:
      n_side
        Number of detector strings per side
      dist
        Spacing between strings [m]
      n_z
        Number of detector modules per string
      dist_z
        Distance of modules on a string [m]
      baseline_noise_rate
        Baseline noise rate (default 1E-6Hz)
    """
    modules = []
    line_id = 0

    for irow in range(0, n_side):
        i_this_row = 2 * (n_side - 1) - irow
        x_pos = np.linspace(
            -(i_this_row - 1) / 2 * dist, (i_this_row - 1) / 2 * dist, i_this_row
        )
        y_pos = irow * dist * np.sqrt(3) / 2
        for x in x_pos:
            modules += make_line(
                x, y_pos, n_z, dist_z, rng, baseline_noise_rate, line_id
            )
            line_id += 1

        if irow != 0:
            x_pos = np.linspace(
                -(i_this_row - 1) / 2 * dist, (i_this_row - 1) / 2 * dist, i_this_row
            )
            y_pos = -irow * dist * np.sqrt(3) / 2

            for x in x_pos:
                modules += make_line(
                    x, y_pos, n_z, dist_z, rng, baseline_noise_rate, line_id
                )
                line_id += 1

    return modules


def make_triang(
    side_len,
    oms_per_line=20,
    dist_z=50,
    dark_noise_rate=16 * 1e-5,
    rng=np.random.RandomState(0),
    efficiency=0.5,
):

    height = np.sqrt(side_len ** 2 - (side_len / 2) ** 2)

    modules = make_line(
        -side_len / 2,
        -height / 3,
        oms_per_line,
        dist_z,
        rng,
        dark_noise_rate,
        0,
        efficiency=efficiency,
    )
    modules += make_line(
        side_len / 2,
        -height / 3,
        oms_per_line,
        dist_z,
        rng,
        dark_noise_rate,
        1,
        efficiency=efficiency,
    )
    modules += make_line(
        0,
        2 / 3 * height,
        oms_per_line,
        dist_z,
        rng,
        dark_noise_rate,
        2,
        efficiency=efficiency,
    )

    det = Detector(modules)

    return det


def make_rhombus(
    side_len,
    oms_per_line=20,
    dist_z=50,
    dark_noise_rate=16 * 1e-5,
    rng=np.random.RandomState(0),
):

    modules = make_line(
        -side_len / 2, 0, oms_per_line, dist_z, rng, dark_noise_rate, 0, efficiency=0.3
    )
    modules += make_line(
        side_len / 2, 0, oms_per_line, dist_z, rng, dark_noise_rate, 1, efficiency=0.3
    )
    modules += make_line(
        0,
        np.sqrt(3) / 2 * side_len,
        oms_per_line,
        dist_z,
        rng,
        dark_noise_rate,
        2,
        efficiency=0.3,
    )
    modules += make_line(
        0,
        -np.sqrt(3) / 2 * side_len,
        oms_per_line,
        dist_z,
        rng,
        dark_noise_rate,
        3,
        efficiency=0.3,
    )
    det = Detector(modules)

    return det
