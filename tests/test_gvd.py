import sys
sys.path.append("..")

import numpy as np
from pytest import approx

from hebe.detector import detector_from_geo
from hebe.detector.medium import Medium

GVD = detector_from_geo("../resources/geofiles/gvdgeo")

TOL = 1e-5

def test_gvd_medium() -> None:
    assert GVD.medium==Medium.WATER

def test_gvd_nmodules() -> None:
    assert len(GVD.modules)==288

def test_gvd_center() -> None:
    exp = np.array([-4.44089e-16, -6.130898e-15, 2.40795e-14])
    assert np.max(np.abs(exp - GVD.offset)) < TOL
    
def test_gvd_outer_cylinder() -> None:
    exp = np.array([60.0, 540.0])
    assert np.max(np.abs(np.array(GVD.outer_cylinder - exp))) < TOL

def test_gvd_outer_radius() -> None:
    exp = GVD.outer_radius==276.5863337
    assert np.abs(GVD.outer_radius - exp) < TOL