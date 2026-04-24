"""Unit tests for pure helpers in prometheus.utils."""

from itertools import repeat

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# units.py — physical constants and conversion factors
# ---------------------------------------------------------------------------


class TestUnits:
    """Physical constants and unit conversion factors are self-consistent."""

    def test_GeV_to_MeV_value(self):
        from prometheus.utils.units import GeV_to_MeV

        assert GeV_to_MeV == 1e3

    def test_MeV_to_GeV_value(self):
        from prometheus.utils.units import MeV_to_GeV

        assert MeV_to_GeV == 1e-3

    def test_GeV_MeV_inverse(self):
        from prometheus.utils.units import GeV_to_MeV, MeV_to_GeV

        assert GeV_to_MeV * MeV_to_GeV == pytest.approx(1.0)

    def test_speed_of_light(self):
        from prometheus.utils.units import SpeedOfLight

        assert SpeedOfLight == 299_792_458

    def test_s_to_ns(self):
        from prometheus.utils.units import s_to_ns

        assert s_to_ns == 1e9

    def test_km_to_m(self):
        from prometheus.utils.units import km_to_m

        assert km_to_m == 1e3

    def test_m_to_cm(self):
        from prometheus.utils.units import m_to_cm

        assert m_to_cm == 1e2

    def test_cm_to_m_inverse(self):
        from prometheus.utils.units import cm_to_m, m_to_cm

        assert m_to_cm * cm_to_m == pytest.approx(1.0)

    def test_km_to_cm(self):
        from prometheus.utils.units import km_to_cm

        assert km_to_cm == 1e5


# ---------------------------------------------------------------------------
# iter_or_rep — iterable-or-repeat helper
# ---------------------------------------------------------------------------

from prometheus.utils.iter_or_rep import iter_or_rep


class TestIterOrRep:
    """iter_or_rep returns the sequence as-is, or a repeat for scalars/singletons."""

    def test_multi_element_list_returned_as_is(self):
        result = iter_or_rep([1, 2, 3])
        assert list(result) == [1, 2, 3]

    def test_single_element_list_becomes_repeat(self):
        result = iter_or_rep([42])
        it = iter(result)
        assert next(it) == 42
        assert next(it) == 42  # infinite

    def test_scalar_int_becomes_repeat(self):
        result = iter_or_rep(7)
        it = iter(result)
        assert next(it) == 7
        assert next(it) == 7  # infinite

    def test_multi_element_tuple_returned_as_is(self):
        result = iter_or_rep((10, 20))
        assert list(result) == [10, 20]

    def test_single_element_tuple_becomes_repeat(self):
        result = iter_or_rep((99,))
        it = iter(result)
        assert next(it) == 99
        assert next(it) == 99  # infinite

    def test_existing_repeat_passes_through(self):
        rep = repeat(5)
        result = iter_or_rep(rep)
        assert result is rep

    def test_values_consumed_correctly_from_list(self):
        values = list(iter_or_rep([10, 20, 30]))
        assert values == [10, 20, 30]


# ---------------------------------------------------------------------------
# convert_loss_name — PROPOSAL → PPC name mapping
# ---------------------------------------------------------------------------

from prometheus.utils.convert_loss_name import convert_loss_name


class TestConvertLossName:
    """All valid PROPOSAL loss-type strings map to the expected PPC names."""

    @pytest.mark.parametrize(
        "inp,expected",
        [
            ("epair", "epair"),
            ("brems", "brems"),
            ("photo", "hadr"),
            ("hadr", "hadr"),
            ("ioniz", "delta"),
            ("continuous", "delta"),
        ],
    )
    def test_valid_conversion(self, inp, expected):
        assert convert_loss_name(inp) == expected

    def test_unknown_type_raises(self):
        with pytest.raises(Exception):
            convert_loss_name("not_a_real_type")


# ---------------------------------------------------------------------------
# translators — PDG ↔ f2k / pstring / int-type dictionaries
# ---------------------------------------------------------------------------

from prometheus.utils.translators import (
    PDG_to_f2k,
    PDG_to_pstring,
    f2k_to_PDG,
    int_type_to_str,
    pstring_to_PDG,
)


class TestTranslators:
    """Translation dictionaries contain known physics-correct mappings."""

    def test_muon_pdg_to_f2k(self):
        assert PDG_to_f2k[13] == "mu-"

    def test_antimuon_pdg_to_f2k(self):
        assert PDG_to_f2k[-13] == "mu+"

    def test_electron_pdg_to_f2k(self):
        assert PDG_to_f2k[11] == "e-"

    def test_f2k_to_pdg_muon(self):
        assert f2k_to_PDG["mu-"] == 13

    def test_f2k_to_pdg_antimuon(self):
        assert f2k_to_PDG["mu+"] == -13

    def test_pdg_to_pstring_muon(self):
        assert PDG_to_pstring[13] == "MuMinus"

    def test_pdg_to_pstring_antimuon(self):
        assert PDG_to_pstring[-13] == "MuPlus"

    def test_pdg_to_pstring_numu(self):
        assert PDG_to_pstring[14] == "NuMu"

    def test_pstring_to_pdg_muon(self):
        assert pstring_to_PDG["MuMinus"] == 13

    def test_pstring_to_pdg_numu(self):
        assert pstring_to_PDG["NuMu"] == 14

    def test_int_type_brems(self):
        assert int_type_to_str[1000000002] == "brems"

    def test_int_type_delta(self):
        assert int_type_to_str[1000000003] == "delta"

    def test_int_type_epair(self):
        assert int_type_to_str[1000000004] == "epair"

    def test_unique_f2k_keys_roundtrip(self):
        """For the unique f2k entries, converting back gives the original PDG code."""
        # Build a set of (pdg, f2k) pairs where the f2k key is unique in PDG_to_f2k
        from collections import Counter

        f2k_counts = Counter(PDG_to_f2k.values())
        for pdg, f2k in PDG_to_f2k.items():
            if f2k_counts[f2k] == 1:
                assert f2k_to_PDG[f2k] == pdg


# ---------------------------------------------------------------------------
# path_length_sampling — exponential sampling for hadronic path lengths
# ---------------------------------------------------------------------------

from prometheus.utils.path_length_sampling import path_length_sampling


class TestPathLengthSampling:
    """path_length_sampling returns positive distances for all supported PDG codes."""

    @pytest.mark.parametrize("pdg_id", [2212, 2112])
    def test_nucleon_returns_positive(self, pdg_id):
        result = path_length_sampling(1e3, pdg_id=pdg_id)
        assert result > 0

    @pytest.mark.parametrize("pdg_id", [211, 111, -211, 130, 310, 311, 321, -321])
    def test_pion_kaon_returns_positive(self, pdg_id):
        result = path_length_sampling(1e3, pdg_id=pdg_id)
        assert result > 0

    def test_unknown_pdg_returns_positive(self):
        # Unknown PDG falls through to pion assumption and should still return a value.
        result = path_length_sampling(1e3, pdg_id=99999)
        assert result > 0

    def test_array_input_shape_preserved(self):
        E = np.array([1e2, 1e3, 1e4])
        result = path_length_sampling(E, pdg_id=211)
        assert result.shape == E.shape
        assert np.all(result > 0)

    def test_proton_mean_scale_reasonable(self):
        """The proton mean free path in water-ice should be ~90 cm."""
        rng = np.random.default_rng(0)
        np.random.seed(0)
        samples = [path_length_sampling(1e3, pdg_id=2212) for _ in range(2000)]
        mean = np.mean(samples)
        dens_water = 0.918
        expected_mean = 83.2 / dens_water
        assert abs(mean - expected_mean) / expected_mean < 0.1


# ---------------------------------------------------------------------------
# find_cog — charge-weighted centre-of-gravity
# ---------------------------------------------------------------------------

from prometheus.utils.find_cog import find_cog


class _FakeModule:
    """Minimal stand-in for prometheus.detector.module.Module."""

    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)


class _FakeDetector:
    """Minimal stand-in for a Detector that supports key-lookup."""

    def __init__(self, modules_by_key):
        self._mods = modules_by_key

    def __getitem__(self, key):
        return self._mods[key]


class TestFindCog:
    """find_cog correctly weights positions by hit count."""

    def test_equal_charges_gives_mean(self):
        det = _FakeDetector(
            {
                (0, 0): _FakeModule([0.0, 0.0, 0.0]),
                (0, 1): _FakeModule([1.0, 0.0, 0.0]),
                (0, 2): _FakeModule([2.0, 0.0, 0.0]),
            }
        )
        event = {(0, 0): [1.0] * 3, (0, 1): [1.0] * 3, (0, 2): [1.0] * 3}
        cog = find_cog(event, det)
        np.testing.assert_allclose(cog, [1.0, 0.0, 0.0])

    def test_all_weight_on_one_module(self):
        det = _FakeDetector(
            {
                (0, 0): _FakeModule([0.0, 0.0, 0.0]),
                (0, 1): _FakeModule([10.0, 0.0, 0.0]),
            }
        )
        # Only module (0,1) has hits
        event = {(0, 0): [], (0, 1): [1.0, 2.0, 3.0]}
        cog = find_cog(event, det)
        np.testing.assert_allclose(cog, [10.0, 0.0, 0.0])

    def test_weighted_average_correct(self):
        det = _FakeDetector(
            {
                (0, 0): _FakeModule([0.0, 0.0, 0.0]),
                (0, 1): _FakeModule([4.0, 0.0, 0.0]),
            }
        )
        # 1 hit at 0, 3 hits at 4 → w-avg = (0*1 + 4*3) / 4 = 3
        event = {(0, 0): [1.0], (0, 1): [1.0, 2.0, 3.0]}
        cog = find_cog(event, det)
        np.testing.assert_allclose(cog[0], 3.0)


# ---------------------------------------------------------------------------
# ExtendedEnum — list() helper
# ---------------------------------------------------------------------------

from prometheus.detector.medium import Medium


class TestExtendedEnum:
    """ExtendedEnum.list() returns a list of member name strings."""

    def test_list_returns_list(self):
        assert isinstance(Medium.list(), list)

    def test_list_contains_water(self):
        assert "WATER" in Medium.list()

    def test_list_contains_ice(self):
        assert "ICE" in Medium.list()

    def test_list_elements_are_strings(self):
        for name in Medium.list():
            assert isinstance(name, str)
