"""Tests for the direction conventions in f2k files."""

import numpy as np
import pytest

from prometheus.particle import PropagatableParticle
from prometheus.utils import serialize_to_f2k
from prometheus.utils.write_to_f2k import f2k_angles

_DIRECTIONS = [
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.48, 0.6, 0.64),
    (-0.48, 0.6, -0.64),
]


def _vector(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_official_ppc_recovers_direction(direction):
    theta, phi = f2k_angles(np.array(direction), official_ppc=True)
    # As read in the official f2k.cxx.
    theta = np.radians(180 - theta)
    phi = np.radians(phi + 180 if phi < 180 else phi - 180)
    np.testing.assert_allclose(_vector(theta, phi), direction, atol=1e-12)


@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_bundled_ppc_recovers_direction(direction):
    np.testing.assert_allclose(_vector(*f2k_angles(np.array(direction))), direction, atol=1e-12)


def test_file_header_uses_official_angles(tmp_path):
    particle = PropagatableParticle(211, 1.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0, None, 0.0)
    path = tmp_path / "f2k"
    for official, theta in ((True, 180.0), (False, 0.0)):
        serialize_to_f2k(particle, str(path), official_ppc=official)
        header = path.read_text().splitlines()[1].split()
        assert float(header[header.index("theta") + 1]) == pytest.approx(theta)
