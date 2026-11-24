"""Tests for the new shape and multi-PMT fields on Module (Step 1)."""

import numpy as np
import pytest

from prometheus.detector.module import _OMR, Module

_POS = np.zeros(3)
_KEY = (0, 0)


class TestModuleShapeDefaults:
    def test_default_is_spherical(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.Rr == pytest.approx(_OMR)
        assert m.Rz == pytest.approx(_OMR)

    def test_default_module_type_is_legacy(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.module_type == -1

    def test_default_n_pmts_is_one(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.n_pmts == 1

    def test_default_pmt_dirs_is_single_downward(self):
        m = Module(pos=_POS, key=_KEY)
        assert len(m.pmt_dirs) == 1
        zenith, azimuth = m.pmt_dirs[0]
        assert zenith == pytest.approx(180.0)
        assert azimuth == pytest.approx(0.0)

    def test_default_cable_azimuth_is_none(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.cable_azimuth is None

    def test_default_beta(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.beta == pytest.approx(0.49)

    def test_default_area(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.area == pytest.approx(1.0)


class TestModuleSpheroidGeometry:
    def test_spheroid_stores_rr(self):
        m = Module(pos=_POS, key=_KEY, Rr=0.150, Rz=0.267, module_type=120)
        assert m.Rr == pytest.approx(0.150)

    def test_spheroid_stores_rz(self):
        m = Module(pos=_POS, key=_KEY, Rr=0.150, Rz=0.267, module_type=120)
        assert m.Rz == pytest.approx(0.267)

    def test_spheroid_aspect_ratio(self):
        m = Module(pos=_POS, key=_KEY, Rr=0.150, Rz=0.267, module_type=120)
        assert m.Rz / m.Rr == pytest.approx(0.267 / 0.150)

    def test_spheroid_module_type_stored(self):
        m = Module(pos=_POS, key=_KEY, Rr=0.150, Rz=0.267, module_type=120)
        assert m.module_type == 120


class TestModuleCylindricalGeometry:
    def test_cylindrical_negative_rz(self):
        m = Module(pos=_POS, key=_KEY, Rr=0.06, Rz=-0.38, module_type=200)
        assert m.Rz < 0

    def test_cylindrical_stores_rr(self):
        m = Module(pos=_POS, key=_KEY, Rr=0.06, Rz=-0.38, module_type=200)
        assert m.Rr == pytest.approx(0.06)

    def test_cylindrical_half_height(self):
        m = Module(pos=_POS, key=_KEY, Rr=0.06, Rz=-0.38, module_type=200)
        assert abs(m.Rz) == pytest.approx(0.38)


class TestModuleMultiPMT:
    def test_multi_pmt_dirs_stored(self):
        dirs = [(180.0, 0.0), (0.0, 0.0)]
        m = Module(pos=_POS, key=_KEY, n_pmts=2, pmt_dirs=dirs, module_type=120)
        assert m.pmt_dirs == dirs

    def test_multi_pmt_count(self):
        dirs = [(180.0, 0.0), (0.0, 0.0)]
        m = Module(pos=_POS, key=_KEY, n_pmts=2, pmt_dirs=dirs, module_type=120)
        assert m.n_pmts == 2

    def test_pmt_dirs_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_pmts"):
            Module(pos=_POS, key=_KEY, n_pmts=2, pmt_dirs=[(180.0, 0.0)], module_type=120)

    def test_cable_azimuth_stored(self):
        m = Module(pos=_POS, key=_KEY, cable_azimuth=90.0)
        assert m.cable_azimuth == pytest.approx(90.0)


class TestModuleLegacyRegression:
    """Ensure existing Module usage is completely unaffected."""

    def test_pos_stored(self):
        pos = np.array([1.0, 2.0, 3.0])
        m = Module(pos=pos, key=(0, 0))
        np.testing.assert_array_equal(m.pos, pos)

    def test_key_stored(self):
        m = Module(pos=_POS, key=(3, 7))
        assert m.key == (3, 7)

    def test_default_efficiency(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.efficiency == pytest.approx(0.2)

    def test_default_noise_rate(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.noise_rate == pytest.approx(1e3)

    def test_default_serial_no_is_none(self):
        m = Module(pos=_POS, key=_KEY)
        assert m.serial_no is None

    def test_repr_does_not_raise(self):
        m = Module(pos=np.array([1.0, 2.0, 3.0]), key=(1, 2))
        _ = repr(m)
