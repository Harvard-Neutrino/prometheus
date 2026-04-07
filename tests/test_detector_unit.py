"""Unit tests for the detector subsystem: Module, Medium, Detector, detector_factory."""
import numpy as np
import pytest

from prometheus.detector.module import Module
from prometheus.detector.medium import Medium
from prometheus.detector.detector import (
    Detector,
    IncompatibleSerialNumbersError,
    IncompatibleMACIDsError,
)
from prometheus.detector.detector_factory import (
    detector_from_geo,
    make_line,
    make_grid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _modules_line(n, medium=Medium.WATER):
    """Return n modules evenly spaced on the z-axis and a Detector wrapping them."""
    mods = [
        Module(pos=np.array([0., 0., float(i) * 10.]), key=(0, i))
        for i in range(n)
    ]
    det = Detector(mods, medium)
    return mods, det


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class TestModule:
    def test_pos_stored(self):
        pos = np.array([1., 2., 3.])
        m = Module(pos=pos, key=(0, 0))
        np.testing.assert_array_equal(m.pos, pos)

    def test_key_stored(self):
        m = Module(pos=np.zeros(3), key=(3, 7))
        assert m.key == (3, 7)

    def test_default_efficiency(self):
        m = Module(pos=np.zeros(3), key=(0, 0))
        assert m.efficiency == pytest.approx(0.2)

    def test_default_noise_rate(self):
        m = Module(pos=np.zeros(3), key=(0, 0))
        assert m.noise_rate == pytest.approx(1e3)

    def test_default_serial_no_is_none(self):
        m = Module(pos=np.zeros(3), key=(0, 0))
        assert m.serial_no is None

    def test_custom_efficiency(self):
        m = Module(pos=np.zeros(3), key=(0, 0), efficiency=0.35)
        assert m.efficiency == pytest.approx(0.35)

    def test_repr_does_not_raise(self):
        m = Module(pos=np.array([1., 2., 3.]), key=(1, 2))
        _ = repr(m)


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------

class TestMedium:
    def test_water_exists(self):
        assert Medium.WATER

    def test_ice_exists(self):
        assert Medium.ICE

    def test_water_value(self):
        assert Medium.WATER.value == 1

    def test_ice_value(self):
        assert Medium.ICE.value == 2

    def test_list_returns_both(self):
        names = Medium.list()
        assert "WATER" in names
        assert "ICE" in names


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------

class TestDetectorConstruction:
    def test_n_modules(self):
        _, det = _modules_line(5)
        assert det.n_modules == 5

    def test_modules_property_length(self):
        _, det = _modules_line(4)
        assert len(det.modules) == 4

    def test_medium_property(self):
        _, det = _modules_line(3, medium=Medium.ICE)
        assert det.medium == Medium.ICE

    def test_offset_is_mean_position(self):
        mods = [
            Module(pos=np.array([0., 0., 0.]), key=(0, 0)),
            Module(pos=np.array([2., 0., 0.]), key=(0, 1)),
        ]
        det = Detector(mods, Medium.WATER)
        np.testing.assert_allclose(det.offset, [1., 0., 0.])

    def test_outer_radius_positive(self):
        _, det = _modules_line(5)
        assert det.outer_radius > 0

    def test_outer_cylinder_is_two_tuple(self):
        _, det = _modules_line(5)
        r, h = det.outer_cylinder
        assert r >= 0
        assert h > 0

    def test_outer_cylinder_height_matches_z_span(self):
        _, det = _modules_line(5)
        z_vals = [0., 10., 20., 30., 40.]
        expected_h = max(z_vals) - min(z_vals)
        _, actual_h = det.outer_cylinder
        assert actual_h == pytest.approx(expected_h)

    def test_module_coords_shape(self):
        _, det = _modules_line(6)
        assert det.module_coords.shape == (6, 3)

    def test_module_efficiencies_shape(self):
        _, det = _modules_line(4)
        assert det.module_efficiencies.shape == (4,)

    def test_module_noise_rates_shape(self):
        _, det = _modules_line(4)
        assert det.module_noise_rates.shape == (4,)


# ---------------------------------------------------------------------------
# Detector.__getitem__
# ---------------------------------------------------------------------------

class TestDetectorGetItem:
    def test_lookup_returns_correct_module(self):
        mods, det = _modules_line(3)
        retrieved = det[(0, 1)]
        assert retrieved is mods[1]

    def test_lookup_position_correct(self):
        _, det = _modules_line(3)
        module = det[(0, 2)]
        np.testing.assert_allclose(module.pos, [0., 0., 20.])

    def test_lookup_missing_key_raises(self):
        _, det = _modules_line(3)
        with pytest.raises(ValueError):
            _ = det[(99, 99)]


# ---------------------------------------------------------------------------
# Detector.__add__
# ---------------------------------------------------------------------------

class TestDetectorAdd:
    def test_add_same_medium_combines_modules(self):
        _, det1 = _modules_line(2)
        mods2 = [Module(pos=np.array([100., 0., float(i) * 10.]), key=(1, i)) for i in range(3)]
        det2 = Detector(mods2, Medium.WATER)
        combined = det1 + det2
        assert combined.n_modules == 5

    def test_add_incompatible_media_raises(self):
        _, det_water = _modules_line(2, medium=Medium.WATER)
        mods_ice = [Module(pos=np.array([0., 0., float(i)]), key=(1, i)) for i in range(2)]
        det_ice = Detector(mods_ice, Medium.ICE)
        with pytest.raises(ValueError):
            _ = det_water + det_ice

    def test_add_preserves_medium(self):
        _, det1 = _modules_line(2, Medium.ICE)
        mods2 = [Module(pos=np.array([10., 0., float(i)]), key=(1, i)) for i in range(2)]
        det2 = Detector(mods2, Medium.ICE)
        combined = det1 + det2
        assert combined.medium == Medium.ICE


# ---------------------------------------------------------------------------
# make_line factory
# ---------------------------------------------------------------------------

class TestMakeLine:
    def test_returns_correct_number_of_modules(self):
        mods = make_line(x=0., y=0., n_z=10, dist_z=5., z_cent=0., line_id=0, rng=42)
        assert len(mods) == 10

    def test_all_modules_share_xy(self):
        mods = make_line(x=3., y=-2., n_z=5, dist_z=10., z_cent=0., line_id=0, rng=0)
        for m in mods:
            assert m.pos[0] == pytest.approx(3.)
            assert m.pos[1] == pytest.approx(-2.)

    def test_line_id_used_as_string_key(self):
        mods = make_line(x=0., y=0., n_z=4, dist_z=5., z_cent=0., line_id=7, rng=0)
        for m in mods:
            assert m.key[0] == 7

    def test_module_indices_sequential(self):
        mods = make_line(x=0., y=0., n_z=4, dist_z=5., z_cent=0., line_id=0, rng=0)
        indices = [m.key[1] for m in mods]
        assert indices == list(range(4))

    def test_returns_list_of_modules(self):
        mods = make_line(x=0., y=0., n_z=3, dist_z=5., z_cent=0., line_id=0, rng=0)
        for m in mods:
            assert isinstance(m, Module)


# ---------------------------------------------------------------------------
# make_grid factory
# ---------------------------------------------------------------------------

class TestMakeGrid:
    def test_returns_detector(self):
        det = make_grid(n_side=2, dist=50., n_z=3, dist_z=10., z_cent=0., medium=Medium.WATER, rng=1)
        assert isinstance(det, Detector)

    def test_n_modules_correct(self):
        # n_side=2 → 4 strings; n_z=3 → 12 modules total
        det = make_grid(n_side=2, dist=50., n_z=3, dist_z=10., z_cent=0., medium=Medium.WATER, rng=1)
        assert det.n_modules == 2 * 2 * 3

    def test_medium_passed_through(self):
        det = make_grid(n_side=2, dist=50., n_z=3, dist_z=10., z_cent=0., medium=Medium.ICE, rng=1)
        assert det.medium == Medium.ICE

    def test_deterministic_with_same_seed(self):
        det1 = make_grid(n_side=2, dist=50., n_z=3, dist_z=10., z_cent=0., medium=Medium.WATER, rng=42)
        det2 = make_grid(n_side=2, dist=50., n_z=3, dist_z=10., z_cent=0., medium=Medium.WATER, rng=42)
        np.testing.assert_allclose(det1.module_coords, det2.module_coords)


# ---------------------------------------------------------------------------
# detector_from_geo factory
# ---------------------------------------------------------------------------

WATER_GEO = "resources/geofiles/demo_water.geo"
ICE_GEO   = "resources/geofiles/demo_ice.geo"


class TestDetectorFromGeo:
    def test_water_geo_returns_detector(self):
        det = detector_from_geo(WATER_GEO)
        assert isinstance(det, Detector)

    def test_water_geo_medium_is_water(self):
        det = detector_from_geo(WATER_GEO)
        assert det.medium == Medium.WATER

    def test_water_geo_has_modules(self):
        det = detector_from_geo(WATER_GEO)
        assert det.n_modules > 0

    def test_ice_geo_returns_detector(self):
        det = detector_from_geo(ICE_GEO)
        assert isinstance(det, Detector)

    def test_ice_geo_medium_is_ice(self):
        det = detector_from_geo(ICE_GEO)
        assert det.medium == Medium.ICE

    def test_custom_efficiency_applied(self):
        det = detector_from_geo(WATER_GEO, efficiency=0.5)
        assert np.all(det.module_efficiencies == pytest.approx(0.5))

    def test_module_positions_are_3d(self):
        det = detector_from_geo(WATER_GEO)
        assert det.module_coords.shape[1] == 3

    def test_outer_radius_positive(self):
        det = detector_from_geo(WATER_GEO)
        assert det.outer_radius > 0
