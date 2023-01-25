import sys
sys.path.append("..")

import numpy as np
from pytest import approx

from prometheus.detector import detector_from_geo
from prometheus.detector.medium import Medium

ICECUBE = detector_from_geo("../prometheus/data/icecube-geo")

TOL = 1e-5

def test_icecube_medium() -> None:
    assert ICECUBE.medium==Medium.ICE

def test_icecube_nmodules() -> None:
    assert len(ICECUBE.modules)==5160

def test_icecube_center() -> None:
    exp = np.array([5.87082946, -2.51860853, -1971.9757655])
    assert np.max(np.abs(exp - ICECUBE.offset)) < TOL
    
def test_icecube_outer_cylinder() -> None:
    exp = np.array([601.1788613216536, 2847.02])
    assert np.max(np.abs(np.array(ICECUBE.outer_cylinder - exp))) < TOL

def test_icecube_outer_radius() -> None:
    assert ICECUBE.outer_radius==2530.6934001968707
