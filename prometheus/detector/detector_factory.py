import numpy as np
import scipy
from typing import Union, List
import itertools

from .module import Module
from .medium import Medium
from .detector import Detector
from .utils import random_serial
from ..utils import iter_or_rep

class InvalidRNGError(Exception):
    """Raised when rng specification can't be parsed"""
    def __init__(self, rng):
        self.message = f"Unable to determine random state seeding from {rng}"
        super.__init__(self.message)

def parse_rng(rng: Union[None, int, np.random.RandomState]) -> np.random.RandomState:
    """Helps determine random number generation state from input

    params
    ______
    rng: rng generator to make sense of

    returns
    _______
    rng: np.random.RandomState

    raises
    ______
    InvalidRNGError: If we don't know how to handle the input rng
    """
    if not (
        isinstance(rng, int) or \
        isinstance(rng, np.random.RandomState) or \
        rng is None
    ):
        raise InvalidRNGError(rng)
    if rng is None or isinstance(rng, int):
        rng = np.random.RandomState(rng)
    return rng

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
    if medium_string and medium_string not in Medium.list():
        raise ValueError(f"Unknown medium. Only {Medium.list()} currently supported")
    if not medium_string:
        return None
    return getattr(Medium, medium_string)

def detector_from_geo(
    geofile: str,
    efficiency: float=0.2,
    noise_rate: float=1
) -> Detector:
    """Make a detector from a Prometheus geo file
    
    params
    ______
    geofile: Geofile to read from
    efficiency: Quantum efficiency of OMs
    noise_rate: Noise rate of OMs in Hz

    returns
    _______
    detector: Prometheus detector object
    """
    pos = []
    keys = []
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

def make_line(
    x: float,
    y: float,
    n_z: int,
    dist_z: float,
    z_cent: float,
    line_id: int,
    rng: np.random.RandomState = 1337,
    baseline_noise_rate: float = 1.0e3,
    efficiency: float = 0.2
) -> List[Module]:
    """Make a line of detector modules. The modules share the same (x, y) coordinate and 
    are spaced along the z-direction. This detector will be symetrically spaced about z=z_cent

    params
    ______
    x: x-position of the line
    y: y-position of the line
    n_z: number of modules on the line
    dist_z: vertical spacing between modules
    z_cent: z-position of the center of the line
    line_id: integer identifier of the line
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs

    returns
    _______
    modules: list of modules. This may be then be used to construct a Prometheus
        Detector if a single line detector is desired
    """
    baseline_noise_rate *= 1e9
    rng = parse_rng(rng)
    modules = []
    zmin = -dist_z * n_z / 2 + z_cent
    zmax = dist_z * n_z / 2 + z_cent
    for idx, z in enumerate(np.linspace(zmin, zmax, n_z)):
        pos = np.array([x, y, z])
        noise_rate = (
            scipy.stats.gamma.rvs(1, 0.25, random_state=rng) * baseline_noise_rate
        )
        mod = Module(
            pos,
            key=(line_id, idx),
            noise_rate=noise_rate,
            efficiency=efficiency
        )
        modules.append(mod)
    return modules


def make_grid(
    n_side: int,
    dist: float,
    n_z: int,
    dist_z: float,
    z_cent: float,
    medium: Medium,
    rng: Union[int, None, np.random.RandomState] = 1337,
    baseline_noise_rate: float = 1.0e3,
    efficiency: float = 0.2
) -> Detector:
    """Build a square detector grid. Strings of detector modules are placed 
    on a square grid, with the number of strings per side, number of modules 
    per string, and z-spacing on a string set by input. The noise rate for 
    each module is randomly sampled from a gamma distribution. The random
    state may be set by input

    params
    ______
    n_side: number of strings per side
    dist: spacing between strings along principal axes
    n_z: number of modules on the line
    dist_z: vertical spacing between modules
    z_cent: z-position of the center of the detector
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs

    returns
    _______
    det: Orthogonal Prometheus Detector
    """
    modules = []
    x_pos = np.linspace(-n_side / 2 * dist, n_side / 2 * dist, n_side)
    y_pos = x_pos

    for idx, (x, y) in enumerate(itertools.product(x_pos, y_pos)):
        modules += make_line(
            x,
            y,
            n_z,
            dist_z,
            z_cent,
            idx,
            rng=rng,
            baseline_noise_rate=baseline_noise_rate,
            efficiency=efficiency
        )

    det = Detector(modules, medium)
    return det


def make_hex_grid(
    n_side: int,
    dist: float,
    n_z: int,
    dist_z: float,
    z_cent: float,
    medium: Medium, 
    baseline_noise_rate: float = 1e3,
    rng: Union[int, None, np.random.RandomState] = 1337,
    efficiency: float = 0.2
) -> Detector:
    """Build a hex detector grid. Strings of detector modules are placed on a hexagonal
    grid with number of OMs per string and distance between these modules set by input.
    The vertical center of the detector is at z_cent.
    The noise rate for each module is randomöy sampled from a gamma distribution.

    params
    ______
    n_side: number of strings per side of hexagon
    dist: insterstring spacing in meters
    n_z: number of modules on the line
    dist_z: vertical spacing between modules
    z_cent: z-position of the center of the line
    line_id: integer identifier of the line
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs

    returns
    _______
    det: Hexagonal Prometheus detector
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
                x,
                y_pos,
                n_z,
                dist_z,
                z_cent,
                line_id,
                rng=rng,
                baseline_noise_rate=baseline_noise_rate,
                efficiency=efficiency
            )
            line_id += 1

        if irow==0:
            continue

        x_pos = np.linspace(
            -(i_this_row - 1) / 2 * dist,
            (i_this_row - 1) / 2 * dist,
            i_this_row
        )
        y_pos = -irow * dist * np.sqrt(3) / 2

        for x in x_pos:
            modules += make_line(
                x,
                y_pos,
                n_z,
                dist_z,
                z_cent,
                line_id,
                rng=rng,
                baseline_noise_rate=baseline_noise_rate,
                efficiency=efficiency
            )
            line_id += 1

    det = Detector(modules, medium)
    return det

def make_triang(
    side_len,
    n_z,
    dist_z,
    z_cent,
    medium: Medium,
    rng: np.random.RandomState = 1337,
    baseline_noise_rate: float = 1.0e3,
    efficiency: float = 0.2
) -> Detector:
    """Build a triangular detector grid. Strings of detector modules are placed 
    on a the corners of a equilateral triangle, with input side length,
    number of modules per string, and z-spacing on a string set by input.
    The noise rate for each module is randomly sampled from a gamma distribution. 
    The random state may be set by input

    params
    ______
    side_len: length of the triangle in meters
    n_z: number of modules on the line
    dist_z: vertical spacing between modules
    z_cent: z-position of the center of the detector
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs

    returns
    _______
    det: a triangular detector
    """

    height = np.sqrt(side_len ** 2 - (side_len / 2) ** 2)

    modules = make_line(
        -side_len / 2,
        -height / 3,
        n_z,
        dist_z,
        z_cent,
        0,
        rng=rng,
        baseline_noise_rate=baseline_noise_rate,
        efficiency=efficiency
    )
    modules += make_line(
        side_len / 2,
        -height / 3,
        n_z,
        dist_z,
        z_cent,
        1,
        rng=rng,
        baseline_noise_rate=baseline_noise_rate,
        efficiency=efficiency
    )
    modules += make_line(
        0,
        2 / 3 * height,
        n_z,
        dist_z,
        z_cent,
        2,
        rng=rng,
        baseline_noise_rate=baseline_noise_rate,
        efficiency=efficiency
    )

    det = Detector(modules, medium)
    return det


def make_rhombus(
    side_len,
    n_z,
    dist_z,
    z_cent,
    medium: Medium,
    rng: Union[int, None, np.random.RandomState] = 1337,
    baseline_noise_rate: float = 1.0e3,
    efficiency: float = 0.2
) -> Detector:
    """Make a rhombus detector

    params
    ______
    side_len: length of the rhombus in meters
    n_z: number of modules on the line
    dist_z: vertical spacing between modules in meters
    z_cent: z-position of the center of the detector in meters
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs

    returns
    _______
    det: A rhombus detector

    """

    modules = make_line(
        -side_len / 2,
        0,
        n_z,
        dist_z,
        z_cent,
        0,
        rng=rng,
        baseline_noise_rate=baseline_noise_rate,
        efficiency=efficiency
    )
    modules += make_line(
        side_len / 2,
        0,
        n_z,
        dist_z,
        z_cent,
        1,
        rng=rng,
        baseline_noise_rate=baseline_noise_rate,
        efficiency=efficiency
    )
    modules += make_line(
        0,
        np.sqrt(3) / 2 * side_len,
        n_z,
        dist_z,
        z_cent,
        2,
        rng=rng,
        baseline_noise_rate=baseline_noise_rate,
        efficiency=efficiency
    )
    modules += make_line(
        0,
        -np.sqrt(3) / 2 * side_len,
        n_z,
        dist_z,
        z_cent,
        3,
        rng=rng,
        baseline_noise_rate=baseline_noise_rate,
        efficiency=efficiency
    )
    det = Detector(modules, medium)
    return det
