"""Tests for parse_ppc (Step 4)."""

import pytest

from prometheus.photon_propagation.utils.parse_ppc import parse_ppc

# Representative HIT lines. Token layout after HIT: string om time wavelength
# pth pph dth dph, where pth/pph (tokens 5,6) are the PHOTON DIRECTION -> photon_*
# and dth/dph (tokens 7,8) are the OM-IMPACT position -> om_*.
_LEGACY_HIT = "HIT 1 42 1234.5 400.0 1.1 2.2 0.5 1.0\n"
_NEXTGEN_HIT_PMT1 = "HIT 1 42_1 1234.5 400.0 1.1 2.2 0.5 1.0\n"
_NEXTGEN_HIT_PMT0 = "HIT 1 42_0 5678.9 420.0 0.9 1.8 0.3 0.7\n"


def _write(tmp_path, lines):
    p = tmp_path / "ppc_out.txt"
    p.write_text("".join(lines))
    return str(p)


# ---------------------------------------------------------------------------
# Empty / non-HIT content
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_file_returns_empty_list(self, tmp_path):
        p = _write(tmp_path, [])
        assert parse_ppc(p) == []

    def test_non_hit_lines_ignored(self, tmp_path):
        lines = [
            "# comment\n",
            "INFO something\n",
            _LEGACY_HIT,
        ]
        hits = parse_ppc(_write(tmp_path, lines))
        assert len(hits) == 1

    def test_only_header_returns_empty(self, tmp_path):
        lines = ["EM 1 1 0 0 0 0\n", "EE\n"]
        assert parse_ppc(_write(tmp_path, lines)) == []


# ---------------------------------------------------------------------------
# Legacy format
# ---------------------------------------------------------------------------


class TestLegacyFormat:
    def test_parse_string_id(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].string_id == 1

    def test_parse_om_id(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].om_id == 42

    def test_parse_time(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].time == pytest.approx(1234.5)

    def test_parse_wavelength(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].wavelength == pytest.approx(400.0)

    def test_parse_om_zenith(self, tmp_path):
        # om_* = OM-impact position = tokens 7,8 = 0.5, 1.0
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].om_zenith == pytest.approx(0.5)

    def test_parse_om_azimuth(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].om_azimuth == pytest.approx(1.0)

    def test_parse_photon_zenith(self, tmp_path):
        # photon_* = photon direction = tokens 5,6 = 1.1, 2.2
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].photon_zenith == pytest.approx(1.1)

    def test_parse_photon_azimuth(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].photon_azimuth == pytest.approx(2.2)

    def test_pmt_id_is_none(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_LEGACY_HIT]))
        assert hits[0].pmt_id is None

    def test_multi_hit_all_pmt_none(self, tmp_path):
        lines = [_LEGACY_HIT, "HIT 2 10 9.9 380.0 0.1 0.2 0.3 0.4\n"]
        hits = parse_ppc(_write(tmp_path, lines))
        assert all(h.pmt_id is None for h in hits)


# ---------------------------------------------------------------------------
# Nextgen format
# ---------------------------------------------------------------------------


class TestNextgenFormat:
    def test_parse_pmt_index_1(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_NEXTGEN_HIT_PMT1]))
        assert hits[0].pmt_id == 1

    def test_parse_pmt_index_0(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_NEXTGEN_HIT_PMT0]))
        assert hits[0].pmt_id == 0

    def test_parse_om_id_from_nextgen(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_NEXTGEN_HIT_PMT1]))
        assert hits[0].om_id == 42

    def test_parse_time_nextgen(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_NEXTGEN_HIT_PMT1]))
        assert hits[0].time == pytest.approx(1234.5)

    def test_nextgen_multiple_hits(self, tmp_path):
        lines = [_NEXTGEN_HIT_PMT1, _NEXTGEN_HIT_PMT0]
        hits = parse_ppc(_write(tmp_path, lines))
        assert len(hits) == 2
        assert hits[0].pmt_id == 1
        assert hits[1].pmt_id == 0

    def test_nextgen_all_fields_populated(self, tmp_path):
        hits = parse_ppc(_write(tmp_path, [_NEXTGEN_HIT_PMT1]))
        h = hits[0]
        assert h.string_id == 1
        assert h.om_id == 42
        assert h.pmt_id == 1
        assert h.time == pytest.approx(1234.5)
        assert h.wavelength == pytest.approx(400.0)
        assert h.om_zenith == pytest.approx(0.5)
        assert h.om_azimuth == pytest.approx(1.0)
        assert h.photon_zenith == pytest.approx(1.1)
        assert h.photon_azimuth == pytest.approx(2.2)


# ---------------------------------------------------------------------------
# Mixed format error
# ---------------------------------------------------------------------------


class TestMixedFormat:
    def test_mixed_format_raises(self, tmp_path):
        lines = [_LEGACY_HIT, _NEXTGEN_HIT_PMT1]
        with pytest.raises(ValueError, match="mixes"):
            parse_ppc(_write(tmp_path, lines))

    def test_nextgen_then_legacy_raises(self, tmp_path):
        lines = [_NEXTGEN_HIT_PMT1, _LEGACY_HIT]
        with pytest.raises(ValueError, match="mixes"):
            parse_ppc(_write(tmp_path, lines))
