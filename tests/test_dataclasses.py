"""Unit tests for domain dataclasses and simple value-objects."""

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Particle / PropagatableParticle
# ---------------------------------------------------------------------------
from prometheus.particle.particle import Particle, PropagatableParticle


def _make_particle(pdg=13, e=1e3, pos=None, direction=None, idx=0):
    if pos is None:
        pos = np.array([0.0, 0.0, 0.0])
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    return Particle(pdg, e, pos, direction, idx)


def _make_propagatable(pdg=13, e=1e3, pos=None, direction=None, idx=0, parent=None):
    if pos is None:
        pos = np.array([0.0, 0.0, 0.0])
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    return PropagatableParticle(pdg, e, pos, direction, idx, parent)


class TestParticle:
    def test_str_muon(self):
        p = _make_particle(pdg=13)
        assert str(p) == "MuMinus"

    def test_str_antimuon(self):
        p = _make_particle(pdg=-13)
        assert str(p) == "MuPlus"

    def test_str_numu(self):
        p = _make_particle(pdg=14)
        assert str(p) == "NuMu"

    def test_int_returns_pdg(self):
        p = _make_particle(pdg=13)
        assert int(p) == 13

    def test_int_antimuon(self):
        p = _make_particle(pdg=-13)
        assert int(p) == -13

    def test_theta_pointing_along_z(self):
        p = _make_particle(direction=np.array([0.0, 0.0, 1.0]))
        assert p.theta == pytest.approx(0.0)

    def test_theta_pointing_against_z(self):
        p = _make_particle(direction=np.array([0.0, 0.0, -1.0]))
        assert p.theta == pytest.approx(math.pi)

    def test_theta_transverse(self):
        p = _make_particle(direction=np.array([1.0, 0.0, 0.0]))
        assert p.theta == pytest.approx(math.pi / 2)

    def test_phi_along_x(self):
        p = _make_particle(direction=np.array([1.0, 0.0, 0.0]))
        assert p.phi == pytest.approx(0.0)

    def test_phi_along_y(self):
        p = _make_particle(direction=np.array([0.0, 1.0, 0.0]))
        assert p.phi == pytest.approx(math.pi / 2)

    def test_phi_along_negative_x(self):
        p = _make_particle(direction=np.array([-1.0, 0.0, 0.0]))
        assert abs(p.phi) == pytest.approx(math.pi)

    def test_energy_stored(self):
        p = _make_particle(e=2345.6)
        assert p.e == 2345.6

    def test_position_stored(self):
        pos = np.array([1.0, 2.0, 3.0])
        p = _make_particle(pos=pos)
        np.testing.assert_array_equal(p.position, pos)


class TestPropagatableParticle:
    def test_default_children_empty(self):
        p = _make_propagatable()
        assert p.children == []

    def test_default_losses_empty(self):
        p = _make_propagatable()
        assert p.losses == []

    def test_default_hits_empty(self):
        p = _make_propagatable()
        assert p.hits == []

    def test_children_list_is_mutable_per_instance(self):
        p1 = _make_propagatable()
        p2 = _make_propagatable()
        p1.children.append("child")
        assert len(p2.children) == 0

    def test_parent_stored(self):
        parent = _make_particle()
        child = _make_propagatable(parent=parent)
        assert child.parent is parent

    def test_add_loss_appended(self):
        p = _make_propagatable()
        p.losses.append("loss_obj")
        assert len(p.losses) == 1

    def test_inherits_particle_theta(self):
        p = _make_propagatable(direction=np.array([0.0, 0.0, 1.0]))
        assert p.theta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

from prometheus.lepton_propagation.loss import Loss


class TestLoss:
    def test_str_brems(self):
        loss = Loss(1000000002, 100.0, np.array([0.0, 0.0, 0.0]))
        assert str(loss) == "brems"

    def test_str_delta(self):
        loss = Loss(1000000003, 50.0, np.array([1.0, 2.0, 3.0]))
        assert str(loss) == "delta"

    def test_str_epair(self):
        loss = Loss(1000000004, 200.0, np.array([0.0, 0.0, 0.0]))
        assert str(loss) == "epair"

    def test_energy_accessible(self):
        loss = Loss(1000000002, 42.0, np.array([0.0, 0.0, 0.0]))
        assert loss.e == 42.0

    def test_position_accessible(self):
        pos = np.array([1.0, 2.0, 3.0])
        loss = Loss(1000000002, 1.0, pos)
        np.testing.assert_array_equal(loss.position, pos)

    def test_frozen_prevents_mutation(self):
        loss = Loss(1000000002, 1.0, np.array([0.0, 0.0, 0.0]))
        with pytest.raises((TypeError, AttributeError)):
            loss.e = 999.0


# ---------------------------------------------------------------------------
# Hit
# ---------------------------------------------------------------------------

from prometheus.photon_propagation.hit import Hit


class TestHit:
    def test_construction_with_all_fields(self):
        h = Hit(
            string_id=1,
            om_id=2,
            time=123.4,
            wavelength=400.0,
            om_zenith=0.5,
            om_azimuth=1.2,
            photon_zenith=0.3,
            photon_azimuth=0.7,
        )
        assert h.string_id == 1
        assert h.om_id == 2
        assert h.time == pytest.approx(123.4)
        assert h.wavelength == pytest.approx(400.0)

    def test_construction_with_none_optionals(self):
        h = Hit(
            string_id=0,
            om_id=0,
            time=0.0,
            wavelength=None,
            om_zenith=None,
            om_azimuth=None,
            photon_zenith=None,
            photon_azimuth=None,
        )
        assert h.wavelength is None
        assert h.om_zenith is None


# ---------------------------------------------------------------------------
# Interactions enum
# ---------------------------------------------------------------------------

from prometheus.injection.interactions import Interactions


class TestInteractions:
    def test_all_four_members_exist(self):
        names = [m.name for m in Interactions]
        assert "GLASHOW_RESONANCE" in names
        assert "CHARGED_CURRENT" in names
        assert "NEUTRAL_CURRENT" in names
        assert "DIMUON" in names

    def test_values_match_spec(self):
        assert Interactions.GLASHOW_RESONANCE.value == 0
        assert Interactions.CHARGED_CURRENT.value == 1
        assert Interactions.NEUTRAL_CURRENT.value == 2
        assert Interactions.DIMUON.value == 3

    def test_enum_members_count(self):
        assert len(list(Interactions)) == 4


# ---------------------------------------------------------------------------
# MCRecord
# ---------------------------------------------------------------------------

from olympus.event_generation.mc_record import MCRecord


class TestMCRecord:
    def test_single_mc_info_wrapped_in_list(self):
        rec = MCRecord("CC", [], {"neutrino_energy": 1e3})
        assert isinstance(rec.mc_info, list)
        assert rec.mc_info[0]["neutrino_energy"] == 1e3

    def test_list_mc_info_not_double_wrapped(self):
        info = [{"a": 1}, {"b": 2}]
        rec = MCRecord("CC", [], info)
        assert len(rec.mc_info) == 2

    def test_event_type_stored(self):
        rec = MCRecord("NC", [], {})
        assert rec.event_type == "NC"

    def test_sources_stored(self):
        rec = MCRecord("CC", ["src1", "src2"], {})
        assert rec.sources == ["src1", "src2"]

    def test_add_combines_records(self):
        r1 = MCRecord("CC", ["s1"], {"e": 1e3})
        r2 = MCRecord("NC", ["s2"], {"e": 2e3})
        combined = r1 + r2
        assert combined.event_type == "CCNC"
        assert combined.sources == ["s1", "s2"]
        assert len(combined.mc_info) == 2

    def test_add_non_mcrecord_raises(self):
        rec = MCRecord("CC", [], {})
        with pytest.raises(NotImplementedError):
            _ = rec + 42


# ---------------------------------------------------------------------------
# PhotonSource / PhotonSourceType
# ---------------------------------------------------------------------------

from olympus.event_generation.photon_source import PhotonSource, PhotonSourceType


class TestPhotonSource:
    def test_default_type_is_standard_cherenkov(self):
        src = PhotonSource(
            position=np.array([0.0, 0.0, 0.0]),
            n_photons=1000,
            time=0.0,
            direction=np.array([0.0, 0.0, 1.0]),
        )
        assert src.type == PhotonSourceType.STANDARD_CHERENKOV

    def test_custom_type_isotropic(self):
        src = PhotonSource(
            position=np.array([0.0, 0.0, 0.0]),
            n_photons=500,
            time=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
            type=PhotonSourceType.ISOTROPIC,
        )
        assert src.type == PhotonSourceType.ISOTROPIC

    def test_attributes_stored_correctly(self):
        pos = np.array([1.0, 2.0, 3.0])
        d = np.array([0.0, 1.0, 0.0])
        src = PhotonSource(position=pos, n_photons=42, time=5.0, direction=d)
        np.testing.assert_array_equal(src.position, pos)
        assert src.n_photons == 42
        assert src.time == pytest.approx(5.0)

    def test_photon_source_type_members(self):
        assert PhotonSourceType.STANDARD_CHERENKOV
        assert PhotonSourceType.ISOTROPIC


# ---------------------------------------------------------------------------
# should_propagate
# ---------------------------------------------------------------------------

from prometheus.photon_propagation.utils.should_propagate import should_propagate


class TestShouldPropagate:
    def test_no_losses_no_children_returns_false(self):
        p = _make_propagatable()
        assert should_propagate(p) is False

    def test_direct_loss_returns_true(self):
        p = _make_propagatable()
        p.losses.append("some_loss")
        assert should_propagate(p) is True

    def test_child_with_loss_returns_true(self):
        parent = _make_propagatable()
        child = _make_propagatable()
        child.losses.append("loss")
        parent.children.append(child)
        assert should_propagate(parent) is True

    def test_child_without_loss_returns_false(self):
        parent = _make_propagatable()
        child = _make_propagatable()
        parent.children.append(child)
        assert should_propagate(parent) is False


# ---------------------------------------------------------------------------
# accumulate_hits
# ---------------------------------------------------------------------------

from prometheus.utils.serialization.accumulate_hits import accumulate_hits


def _make_hit(t=0.0):
    return Hit(
        string_id=0,
        om_id=0,
        time=t,
        wavelength=None,
        om_zenith=None,
        om_azimuth=None,
        photon_zenith=None,
        photon_azimuth=None,
    )


class TestAccumulateHits:
    def test_empty_particle_list(self):
        result = accumulate_hits([])
        assert result == []

    def test_single_particle_no_hits(self):
        p = _make_propagatable(idx=0)
        result = accumulate_hits([p])
        assert result == []

    def test_single_particle_with_hits(self):
        p = _make_propagatable(idx=7)
        p.hits = [_make_hit(1.0), _make_hit(2.0)]
        result = accumulate_hits([p])
        assert len(result) == 2
        for hit, idx in result:
            assert idx == 7

    def test_hit_times_preserved(self):
        p = _make_propagatable(idx=0)
        p.hits = [_make_hit(3.14)]
        result = accumulate_hits([p])
        assert result[0][0].time == pytest.approx(3.14)

    def test_child_hits_collected(self):
        parent = _make_propagatable(idx=1)
        child = _make_propagatable(idx=2)
        child.hits = [_make_hit(5.0)]
        parent.children.append(child)
        result = accumulate_hits([parent])
        assert len(result) == 1
        assert result[0][1] == 2

    def test_nested_children_collected_recursively(self):
        grandparent = _make_propagatable(idx=0)
        parent = _make_propagatable(idx=1)
        child = _make_propagatable(idx=2)
        child.hits = [_make_hit(1.0), _make_hit(2.0)]
        parent.children.append(child)
        grandparent.children.append(parent)
        result = accumulate_hits([grandparent])
        assert len(result) == 2

    def test_multiple_top_level_particles(self):
        p1 = _make_propagatable(idx=0)
        p2 = _make_propagatable(idx=1)
        p1.hits = [_make_hit(1.0)]
        p2.hits = [_make_hit(2.0)]
        result = accumulate_hits([p1, p2])
        assert len(result) == 2
        assert {idx for _, idx in result} == {0, 1}
