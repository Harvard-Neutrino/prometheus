import numpy as np
from typing import Union

from .module import Module
from .medium import Medium
from .detector import Detector
from .utils import random_serial
from ..utils import iter_or_rep


def detector_from_f2k(
    fname,
    efficiency=0.2,
    noise_rate=1
) -> Detector:
    """
    """
    pos = []
    keys = []
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
        for p, k, e, nr, ser in zip(pos, keys, efficiency, noise_rate, sers)
    ]
    det = Detector(modules)
    return det

def read_medium(geofile) -> Union[Medium, None]:
    """Figures out detector medium from geofile

    params
    ______
    geofile: Detector geometry file
    
    returns
    _______
    medium: Medium for the detector
    """
    medium_string = ""
    with open(geofile) as geo_in:
        for line in geo_in.readlines():
            if "medium" in line.lower():
                medium_string = line.split()[1].upper()
                break
    # If a medium was provided, make sure we know how to handle it
    if medium_string and medium_string not in Medium.list():
        raise ValueError(f"Unknown medium. Only {Medium.list()} currently supported")
    # If no medium was specified, return None
    if not medium_string:
        return None
    return getattr(Medium, medium_string)

def detector_from_geo(
    geofile: str,
    efficiency: float=0.2,
    noise_rate: float=1
) -> Detector:
    """Makes a detector from a Prometheus geo file
    
    params
    ______
    geofile: Geofile to read from
    efficiency: Quantum efficiency of OMs
    noise_rate: Noise rate of OMs in Hz

    returns
    _______
    detector: Prometheus detector object
    """
    pos = []; keys = []
    medium = read_medium(geofile)
    with open(geofile) as geo_in:
        read_lines = geo_in.readlines()
        modules_i = read_lines.index("### Modules ###\n")   

        for line in read_lines[modules_i+1:]:
            line = line.strip("\n").split("\t")
            pos.append(
                np.array([
                    float(line[0]),
                    float(line[1]),
                    float(line[2])
                ])
            )
            pos_out = np.array(pos)
            keys.append((int(line[3]),int(line[4])))

    sers = [random_serial() for _ in range(len(pos))]

    efficiency = iter_or_rep(efficiency)
    noise_rate = iter_or_rep(noise_rate)

    modules = [
        Module(p, k, efficiency=e, noise_rate=nr, serial_no=ser) 
        for p, k, e, nr, ser in zip(pos, keys, efficiency, noise_rate, sers)
    ]
    det = Detector(modules, medium)
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

