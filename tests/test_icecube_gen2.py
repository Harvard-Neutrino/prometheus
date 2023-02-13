import sys
sys.path.append("..")

import numpy as np
from pytest import approx

from hebe.detector import detector_from_geo
from hebe.detector.medium import Medium

ICECUBE_GEN2 = detector_from_geo("../resources/geofiles/icecube_gen2.geo")

TOL = 1e-5

def test_icecube_gen2_medium() -> None:
    assert ICECUBE_GEN2.medium==Medium.ICE

def test_icecube_gen2_nmodules() -> None:
    assert len(ICECUBE_GEN2.modules)==14906

def test_icecube_gen2_center() -> None:
    exp = np.array([-462.970565, 186.756135, 0])
    assert np.max(np.abs(exp - ICECUBE_GEN2.offset)) < TOL
    
def test_icecube_gen2_outer_cylinder() -> None:
    exp = np.array([1932.103134, 1280.0])
    assert np.max(np.abs(np.array(ICECUBE_GEN2.outer_cylinder - exp))) < TOL

def test_icecube_gen2_outer_radius() -> None:
    exp =  ICECUBE_GEN2.outer_radius==2035.3433422
    assert np.abs(ICECUBE_GEN2.outer_radius - exp) < TOL
