"""Tests of the parts of scripts/check_lom16_yield.py that do not need PPC."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parents[1]
_DATA = _ROOT / "examples" / "data" / "lom"


@pytest.fixture(scope="module")
def check():
    sys.path.insert(0, str(_ROOT / "scripts"))
    path = _ROOT / "scripts" / "check_lom16_yield.py"
    spec = importlib.util.spec_from_file_location("check_lom16_yield", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sphere_points_are_unit_vectors_without_a_preferred_direction(check):
    points = check.sphere_points(48)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)
    np.testing.assert_allclose(points.mean(axis=0), 0.0, atol=0.02)


def test_dom_orientations_average_to_no_direction(check):
    angles = np.radians(check.FACING)
    directions = check.unit_vector(angles[:, 0], angles[:, 1])
    np.testing.assert_allclose(directions.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(directions.T @ directions, 2 * np.eye(3), atol=1e-12)


def test_unattenuated_yield_agrees_with_fig_76():
    """Fig. 75 weighted with a Cherenkov spectrum gives the start of Fig. 76."""
    areas = []
    for name in ("lom16_wavelength.txt", "gen1_wavelength.txt"):
        x, y = np.loadtxt(_DATA / name, unpack=True)
        wavelength = np.arange(280.0, 681.0, 2.0)
        areas.append(np.sum(np.interp(wavelength, x, y) / wavelength**2))
    distance, ratio = np.loadtxt(_DATA / "lom16_relative_yield.txt", unpack=True)
    assert distance[0] == 0.0
    assert areas[0] / areas[1] == pytest.approx(ratio[0], rel=0.02)
