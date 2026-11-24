"""Tests for scripts/fit_om_conf.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from prometheus.detector import read_om_conf

_SCRIPT = Path(__file__).parents[1] / "scripts" / "fit_om_conf.py"
_SPEC = importlib.util.spec_from_file_location("fit_om_conf", _SCRIPT)
fom = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fom)

_RINGS = ((1, 0.0), (1, 45.0), (-1, 45.0), (-1, 0.0))


def _module(polar, equatorial, beta):
    dirs = []
    for (sign, phase), angle in zip(_RINGS, (polar, equatorial, equatorial, polar)):
        zenith = angle if sign > 0 else 180 - angle
        dirs += [(zenith, phase + 90 * i) for i in range(4)]
    return {
        "module_type": 171,
        "area": 1.0,
        "beta": beta,
        "Rr": 0.156,
        "Rz": 0.222,
        "n_pmts": 16,
        "pmt_dirs": dirs,
        "cable_azimuth": None,
    }


@pytest.mark.parametrize("x", [-1.0, -0.4, 0.0, 0.3, 1.0])
def test_response_limits(x):
    x = np.array([x])
    np.testing.assert_allclose(fom.pmt_response(x, 1.0), np.maximum(x, 0))
    np.testing.assert_allclose(fom.pmt_response(x, 1e-9), (1 + x) / 2, atol=1e-6)


def test_response_peaks_at_one():
    x = np.linspace(-1, 1, 401)
    for beta in (0.2, 0.49, 0.97):
        f = fom.pmt_response(x, beta)
        assert f.max() == pytest.approx(1.0)
        assert f.min() >= -1e-12


def test_recovers_known_module():
    truth = _module(40.0, 70.0, 0.8)
    cos_zenith = np.linspace(-1, 1, 51)
    area = 120 * fom.zenith_profile(cos_zenith, truth["pmt_dirs"], truth["beta"])
    result = fom.fit(cos_zenith, area, _module(36.0, 62.0, 1.0), n_starts=6)
    assert result.rms < 1e-3
    assert result.at_bound == []
    assert result.ring_angles == pytest.approx([40.0, 70.0], abs=1.0)
    assert result.beta == pytest.approx(0.8, abs=0.03)


def test_poles_stay_fixed():
    module = _module(40.0, 70.0, 1.0)
    dirs = module["pmt_dirs"] + [(0.0, 0.0), (180.0, 0.0)]
    angles, index = fom.group_rings(dirs)
    moved = fom.apply_rings(dirs, [50.0, 60.0], index)
    assert angles == [40.0, 70.0]
    assert moved[-2:] == [(0.0, 0.0), (180.0, 0.0)]
    assert moved[0][0] == 50.0
    assert moved[12][0] == 130.0


def test_too_few_points():
    with pytest.raises(ValueError, match="cannot constrain"):
        fom.fit(np.array([0.0, 1.0]), np.array([1.0, 1.0]), _module(40.0, 70.0, 1.0))


def test_output_reads_back(tmp_path):
    module = _module(40.0, 70.0, 0.8)
    result = fom.FitResult(0.8, module["pmt_dirs"], 1.0, 0.0, True, [40.0, 70.0], [])
    path = tmp_path / "om.conf"
    path.write_text(fom.format_module("LOM-16", module, result) + "\n")
    assert read_om_conf(path)[171] == module


def test_equal_rings_keep_their_order():
    _, index = fom.group_rings(_module(40.0, 70.0, 1.0)["pmt_dirs"])
    assert fom.keep_ring_order([36.0, 62.0], [71.0, 45.0], index) == [45.0, 71.0]


def test_reports_parameter_at_limit():
    truth = _module(40.0, 70.0, 0.8)
    cos_zenith = np.linspace(-1, 1, 31)
    area = fom.zenith_profile(cos_zenith, truth["pmt_dirs"], truth["beta"])
    result = fom.fit(cos_zenith, area, _module(36.0, 62.0, 1.0), n_starts=2, max_shift=2.0)
    assert result.at_bound
