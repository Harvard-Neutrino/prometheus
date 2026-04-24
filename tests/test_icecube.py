import numpy as np

from prometheus.detector import detector_from_geo
from prometheus.detector.medium import Medium

ICECUBE = detector_from_geo("resources/geofiles/icecube.geo")

TOL = 1e-5


def test_icecube_medium() -> None:
    assert ICECUBE.medium == Medium.ICE


def test_icecube_nmodules() -> None:
    assert len(ICECUBE.modules) == 5160


def test_icecube_center() -> None:
    exp = np.array([5.87082946, -2.51860853, -1971.9757655])
    assert np.max(np.abs(exp - ICECUBE.offset)) < TOL


def test_icecube_outer_cylinder() -> None:
    exp = np.array([596.280348927972, 1037.3799999999999])
    assert np.max(np.abs(np.array(ICECUBE.outer_cylinder) - exp)) < TOL


def test_icecube_outer_radius() -> None:
    assert abs(ICECUBE.outer_radius - 789.8215434647672) < TOL
