"""Tests for serialize_particles_to_awkward output modes (Step 6)."""

import numpy as np
import pytest

from prometheus.photon_propagation.hit import Hit
from prometheus.utils.serialization.serialize_particles_to_awkward import (
    _VALID_MODES,
    _hit_cartesian,
    serialize_particles_to_awkward,
)

# ---------------------------------------------------------------------------
# Fake detector / injection helpers
# ---------------------------------------------------------------------------


class _FakeModule:
    def __init__(self, key, pos, Rr=0.16510, Rz=0.16510, module_type=-1):
        self.key = key
        self.pos = np.array(pos, dtype=float)
        self.Rr = Rr
        self.Rz = Rz
        self.module_type = module_type


class _FakeDetector:
    def __init__(self, modules):
        self._map = {m.key: m for m in modules}

    def __getitem__(self, key):
        return self._map[key]

    def needs_nextgen(self):
        return any(m.module_type != -1 for m in self._map.values())


class _FakeParticle:
    def __init__(self, hits):
        self.hits = hits
        self.serialization_idx = 0
        self.children = []


class _FakeInjectionEvent:
    def __init__(self, particles):
        self.final_states = particles


class _FakeInjection:
    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)


def _make_hit(
    string_id=1,
    om_id=1,
    time=100.0,
    wavelength=400.0,
    om_zenith=1.0,
    om_azimuth=0.5,
    photon_zenith=0.8,
    photon_azimuth=1.2,
    pmt_id=None,
):
    return Hit(
        string_id=string_id,
        om_id=om_id,
        time=time,
        wavelength=wavelength,
        om_zenith=om_zenith,
        om_azimuth=om_azimuth,
        photon_zenith=photon_zenith,
        photon_azimuth=photon_azimuth,
        pmt_id=pmt_id,
    )


def _simple_setup(Rr=0.16510, Rz=0.16510, module_type=-1, **hit_kwargs):
    mod = _FakeModule(key=(1, 1), pos=[0.0, 0.0, 0.0], Rr=Rr, Rz=Rz, module_type=module_type)
    det = _FakeDetector([mod])
    hit = _make_hit(**hit_kwargs)
    particle = _FakeParticle([hit])
    injection = _FakeInjection([_FakeInjectionEvent([particle])])
    return det, injection


# ---------------------------------------------------------------------------
# Output mode validation
# ---------------------------------------------------------------------------


class TestOutputModeValidation:
    def test_invalid_mode_raises(self):
        det, inj = _simple_setup()
        with pytest.raises(ValueError, match="output_mode"):
            serialize_particles_to_awkward(det, inj, output_mode="bogus")

    def test_valid_modes_do_not_raise(self):
        for mode in _VALID_MODES:
            det, inj = _simple_setup()
            result = serialize_particles_to_awkward(det, inj, output_mode=mode)
            assert result is not None


# ---------------------------------------------------------------------------
# Default mode
# ---------------------------------------------------------------------------


class TestDefaultMode:
    def _get(self):
        det, inj = _simple_setup()
        return serialize_particles_to_awkward(det, inj, output_mode="default")

    def test_has_sensor_pos_x(self):
        assert "sensor_pos_x" in self._get().fields

    def test_has_string_id(self):
        assert "string_id" in self._get().fields

    def test_has_sensor_id(self):
        assert "sensor_id" in self._get().fields

    def test_has_t(self):
        assert "t" in self._get().fields

    def test_has_id_idx(self):
        assert "id_idx" in self._get().fields

    def test_no_wavelength(self):
        assert "wavelength" not in self._get().fields

    def test_no_pmt_id(self):
        assert "pmt_id" not in self._get().fields

    def test_pmt_id_if_the_detector_has_types(self):
        det, inj = _simple_setup(module_type=5, pmt_id=3)
        result = serialize_particles_to_awkward(det, inj)
        assert result["pmt_id"].to_list() == [[3]]
        assert "wavelength" not in result.fields

    def test_no_hit_xyz(self):
        result = self._get()
        assert "hit_x" not in result.fields
        assert "hit_y" not in result.fields
        assert "hit_z" not in result.fields

    def test_time_value_correct(self):
        det, inj = _simple_setup(time=999.9)
        result = serialize_particles_to_awkward(det, inj, output_mode="default")
        import awkward as ak

        assert ak.to_list(result["t"])[0][0] == pytest.approx(999.9)


# ---------------------------------------------------------------------------
# Photon mode
# ---------------------------------------------------------------------------


class TestPhotonMode:
    def _get(self, **kwargs):
        det, inj = _simple_setup(**kwargs)
        return serialize_particles_to_awkward(det, inj, output_mode="photon")

    def test_has_wavelength(self):
        assert "wavelength" in self._get().fields

    def test_has_photon_zenith(self):
        assert "photon_zenith" in self._get().fields

    def test_has_photon_azimuth(self):
        assert "photon_azimuth" in self._get().fields

    def test_has_om_zenith(self):
        assert "om_zenith" in self._get().fields

    def test_has_om_azimuth(self):
        assert "om_azimuth" in self._get().fields

    def test_has_pmt_id(self):
        assert "pmt_id" in self._get().fields

    def test_pmt_id_value_set(self):
        import awkward as ak

        result = self._get(pmt_id=1)
        assert ak.to_list(result["pmt_id"])[0][0] == 1

    def test_pmt_id_none_for_legacy(self):
        import awkward as ak

        result = self._get(pmt_id=None)
        assert ak.to_list(result["pmt_id"])[0][0] is None

    def test_wavelength_value(self):
        import awkward as ak

        result = self._get(wavelength=420.0)
        assert ak.to_list(result["wavelength"])[0][0] == pytest.approx(420.0)

    def test_no_hit_xyz(self):
        result = self._get()
        assert "hit_x" not in result.fields


# ---------------------------------------------------------------------------
# Full mode
# ---------------------------------------------------------------------------


class TestFullMode:
    def _get(self, Rr=0.16510, Rz=0.16510, **kwargs):
        det, inj = _simple_setup(Rr=Rr, Rz=Rz, **kwargs)
        return serialize_particles_to_awkward(det, inj, output_mode="full")

    def test_has_hit_x(self):
        assert "hit_x" in self._get().fields

    def test_has_hit_y(self):
        assert "hit_y" in self._get().fields

    def test_has_hit_z(self):
        assert "hit_z" in self._get().fields

    def test_spherical_impact_on_surface(self):
        import awkward as ak

        R = 0.16510
        result = self._get(Rr=R, Rz=R, om_zenith=np.pi / 3, om_azimuth=np.pi / 4)
        hx = ak.to_list(result["hit_x"])[0][0]
        hy = ak.to_list(result["hit_y"])[0][0]
        hz = ak.to_list(result["hit_z"])[0][0]
        dist = np.sqrt(hx**2 + hy**2 + hz**2)
        assert dist == pytest.approx(R, rel=1e-5)

    def test_spheroid_impact_on_surface(self):
        import awkward as ak

        Rr, Rz = 0.150, 0.267
        om_zenith = np.pi / 4
        om_azimuth = np.pi / 6
        result = self._get(Rr=Rr, Rz=Rz, om_zenith=om_zenith, om_azimuth=om_azimuth)
        hx = ak.to_list(result["hit_x"])[0][0]
        hy = ak.to_list(result["hit_y"])[0][0]
        hz = ak.to_list(result["hit_z"])[0][0]
        on_surf = (hx / Rr) ** 2 + (hy / Rr) ** 2 + (hz / Rz) ** 2
        assert on_surf == pytest.approx(1.0, rel=1e-5)

    def test_cylinder_impact_at_wall(self):
        import awkward as ak

        Rr = 0.06
        result = self._get(Rr=Rr, Rz=-0.38, om_zenith=np.pi / 3, om_azimuth=1.0)
        hx = ak.to_list(result["hit_x"])[0][0]
        hy = ak.to_list(result["hit_y"])[0][0]
        r = np.sqrt(hx**2 + hy**2)
        assert r == pytest.approx(Rr, rel=1e-5)

    def test_cylinder_z_in_range(self):
        import awkward as ak

        Rz = -0.38
        for om_zenith in np.linspace(0.01, np.pi - 0.01, 9):
            result = self._get(Rr=0.06, Rz=Rz, om_zenith=om_zenith, om_azimuth=0.5)
            hz = ak.to_list(result["hit_z"])[0][0]
            assert abs(hz) <= abs(Rz) + 1e-9

    def test_none_angles_produces_none_coords(self):
        import awkward as ak

        det, inj = _simple_setup(om_zenith=None, om_azimuth=None)
        result = serialize_particles_to_awkward(det, inj, output_mode="full")
        hx = ak.to_list(result["hit_x"])[0][0]
        assert hx is None


# ---------------------------------------------------------------------------
# No hits → None
# ---------------------------------------------------------------------------


class TestNoHits:
    def test_returns_none_when_no_hits(self):
        mod = _FakeModule(key=(1, 1), pos=[0.0, 0.0, 0.0])
        det = _FakeDetector([mod])
        particle = _FakeParticle([])
        injection = _FakeInjection([_FakeInjectionEvent([particle])])
        result = serialize_particles_to_awkward(det, injection, output_mode="default")
        assert result is None


# ---------------------------------------------------------------------------
# _hit_cartesian unit tests
# ---------------------------------------------------------------------------


class TestHitCartesian:
    def test_sphere_north_pole(self):
        mod = _FakeModule(key=(1, 1), pos=[0.0, 0.0, 0.0], Rr=0.16510, Rz=0.16510)
        hit = _make_hit(om_zenith=0.0, om_azimuth=0.0)
        hx, hy, hz = _hit_cartesian(hit, mod)
        assert hx == pytest.approx(0.0, abs=1e-10)
        assert hy == pytest.approx(0.0, abs=1e-10)
        assert abs(hz) == pytest.approx(0.16510, rel=1e-5)

    def test_spheroid_equator_z_near_zero(self):
        mod = _FakeModule(key=(1, 1), pos=[0.0, 0.0, 0.0], Rr=0.150, Rz=0.267)
        hit = _make_hit(om_zenith=np.pi / 2, om_azimuth=0.0)
        hx, hy, hz = _hit_cartesian(hit, mod)
        assert hz == pytest.approx(0.0, abs=1e-10)
        assert abs(hx) == pytest.approx(0.150, rel=1e-5)

    def test_spheroid_pole(self):
        mod = _FakeModule(key=(1, 1), pos=[0.0, 0.0, 0.0], Rr=0.150, Rz=0.267)
        hit = _make_hit(om_zenith=0.0, om_azimuth=0.0)
        hx, hy, hz = _hit_cartesian(hit, mod)
        assert abs(hz) == pytest.approx(0.267, rel=1e-5)

    def test_cylinder_wall_radius(self):
        mod = _FakeModule(key=(1, 1), pos=[0.0, 0.0, 0.0], Rr=0.06, Rz=-0.38)
        hit = _make_hit(om_zenith=np.pi / 4, om_azimuth=0.3)
        hx, hy, hz = _hit_cartesian(hit, mod)
        assert np.sqrt(hx**2 + hy**2) == pytest.approx(0.06, rel=1e-5)
