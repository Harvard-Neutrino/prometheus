import sys
sys.path.append("..")

import numpy as np
from pytest import approx

from prometheus.detector import detector_from_geo
from prometheus.detector.medium import Medium

PONE = detector_from_geo("../prometheus/data/pone_triangle-geo")

TOL = 1e-5

def test_pone_medium() -> None:
    assert PONE.medium==Medium.WATER

def test_pone_nmodule() -> None:
    assert len(PONE.modules)==60

def test_pone_center() -> None:
    exp = np.array([0.00000000e+00, 9.00020799e-15, -3.60008319e-14])
    assert np.max(np.abs(exp - PONE.offset)) < TOL

def test_pone_outer_cylinder() -> None:
    exp = np.array([57.735026918962575, 1000.0])
    assert np.max(np.abs(np.array(PONE.outer_cylinder) - exp)) < TOL

def test_pone_outer_radius() -> None:
    exp = 503.32229568471666
    assert np.abs(PONE.outer_radius - exp) < TOL
