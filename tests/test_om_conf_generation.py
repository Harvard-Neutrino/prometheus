"""Tests for Detector.needs_nextgen(), to_om_conf(), and to_om_map() (Step 2)."""

import numpy as np
import pytest

from prometheus.detector.detector import Detector
from prometheus.detector.medium import Medium
from prometheus.detector.module import Module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _legacy_det():
    mods = [Module(pos=np.zeros(3), key=(1, i)) for i in range(3)]
    return Detector(mods, Medium.ICE)


def _degg_mod(key=(1, 1)):
    return Module(
        pos=np.array([0.0, 0.0, 0.0]),
        key=key,
        module_type=120,
        Rr=0.150,
        Rz=0.267,
        beta=0.5,
        area=1.0,
        n_pmts=2,
        pmt_dirs=[(180.0, 0.0), (0.0, 0.0)],
    )


def _wom_mod(key=(1, 2)):
    return Module(
        pos=np.array([0.0, 0.0, 10.0]),
        key=key,
        module_type=200,
        Rr=0.06,
        Rz=-0.38,
        beta=-2.0,
        area=1.0,
        n_pmts=2,
        pmt_dirs=[(180.0, 0.0), (0.0, 0.0)],
    )


def _parse_om_conf(path):
    """Minimal parser that returns {type_id: dict} from an om.conf file."""
    types = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            # If the first non-whitespace is a letter it's a new type header
            if not line[0].isspace():
                if current is not None:
                    types[current["id"]] = current
                name, type_id, area, beta, Rr, Rz, n_pmts, z0, a0 = tokens[:9]
                cable = float(tokens[9]) if len(tokens) > 9 else None
                current = {
                    "id": int(type_id),
                    "area": float(area),
                    "beta": float(beta),
                    "Rr": float(Rr),
                    "Rz": float(Rz),
                    "n_pmts": int(n_pmts),
                    "pmt_dirs": [(float(z0), float(a0))],
                    "cable": cable,
                }
            else:
                # Continuation line: additional PMT direction
                stripped = line.strip()
                if stripped and current is not None:
                    parts = stripped.split()
                    current["pmt_dirs"].append((float(parts[0]), float(parts[1])))
        if current is not None:
            types[current["id"]] = current
    return types


# ---------------------------------------------------------------------------
# needs_nextgen
# ---------------------------------------------------------------------------


class TestNeedsNextgen:
    def test_false_for_all_legacy_modules(self):
        det = _legacy_det()
        assert det.needs_nextgen() is False

    def test_true_when_one_typed_module(self):
        mods = [Module(pos=np.zeros(3), key=(1, 0)), _degg_mod(key=(1, 1))]
        det = Detector(mods, Medium.ICE)
        assert det.needs_nextgen() is True

    def test_true_for_all_typed(self):
        mods = [_degg_mod(key=(1, i)) for i in range(3)]
        det = Detector(mods, Medium.ICE)
        assert det.needs_nextgen() is True


# ---------------------------------------------------------------------------
# to_om_conf
# ---------------------------------------------------------------------------


class TestToOmConf:
    def test_legacy_only_writes_header_only(self, tmp_path):
        det = _legacy_det()
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert len(types) == 0

    def test_degg_rr_preserved(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert types[120]["Rr"] == pytest.approx(0.150)

    def test_degg_rz_preserved(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert types[120]["Rz"] == pytest.approx(0.267)

    def test_degg_n_pmts(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert types[120]["n_pmts"] == 2

    def test_degg_pmt_dir_count(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert len(types[120]["pmt_dirs"]) == 2

    def test_degg_pmt_dirs_correct(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert types[120]["pmt_dirs"][0] == pytest.approx((180.0, 0.0))
        assert types[120]["pmt_dirs"][1] == pytest.approx((0.0, 0.0))

    def test_cable_azimuth_written_when_set(self, tmp_path):
        mod = Module(
            pos=np.zeros(3),
            key=(1, 0),
            module_type=120,
            Rr=0.150,
            Rz=0.267,
            n_pmts=1,
            pmt_dirs=[(180.0, 0.0)],
            cable_azimuth=90.0,
        )
        det = Detector([mod], Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert types[120]["cable"] == pytest.approx(90.0)

    def test_cable_azimuth_omitted_when_none(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert types[120]["cable"] is None

    def test_multiple_types_both_present(self, tmp_path):
        mods = [_degg_mod(key=(1, 0)), _wom_mod(key=(1, 1))]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert 120 in types
        assert 200 in types

    def test_wom_negative_rz(self, tmp_path):
        mods = [_wom_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert types[200]["Rz"] < 0

    def test_duplicate_type_written_once(self, tmp_path):
        mods = [_degg_mod(key=(1, 0)), _degg_mod(key=(1, 1))]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        assert len(types) == 1

    def test_round_trip_aspect_ratio(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf(path)
        F = types[120]["Rz"] / types[120]["Rr"]
        assert F == pytest.approx(0.267 / 0.150, rel=1e-5)


# ---------------------------------------------------------------------------
# to_om_map
# ---------------------------------------------------------------------------


class TestToOmMap:
    def test_legacy_modules_excluded(self, tmp_path):
        det = _legacy_det()
        path = str(tmp_path / "om.map")
        det.to_om_map(path)
        with open(path) as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 0

    def test_typed_module_present(self, tmp_path):
        mods = [Module(pos=np.zeros(3), key=(1, 0)), _degg_mod(key=(1, 1))]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.map")
        det.to_om_map(path)
        with open(path) as f:
            lines = [line.split() for line in f if line.strip()]
        assert len(lines) == 1
        assert int(lines[0][2]) == 120

    def test_correct_string_om_ids(self, tmp_path):
        mods = [_degg_mod(key=(3, 7))]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.map")
        det.to_om_map(path)
        with open(path) as f:
            lines = [line.split() for line in f if line.strip()]
        assert int(lines[0][0]) == 3
        assert int(lines[0][1]) == 7

    def test_type_ids_match_conf(self, tmp_path):
        mods = [_degg_mod(key=(1, 0)), _wom_mod(key=(1, 1))]
        det = Detector(mods, Medium.ICE)
        conf_path = str(tmp_path / "om.conf")
        map_path = str(tmp_path / "om.map")
        det.to_om_conf(conf_path)
        det.to_om_map(map_path)
        conf_types = set(_parse_om_conf(conf_path).keys())
        with open(map_path) as f:
            map_types = {int(line.split()[2]) for line in f if line.strip()}
        assert map_types.issubset(conf_types)
