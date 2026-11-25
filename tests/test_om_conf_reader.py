"""Tests for read_om_conf."""

import numpy as np
import pytest

from prometheus.detector import Detector, Medium, Module, read_om_conf

_CONF = """# name module area beta Rr Rz num dir cable
Single\t7\t1\t0.49\t0.16\t0.16\t1\t180 0\t90
Triple\t8\t0.9\t1\t0.15\t0.25\t3\t40 0
\t\t\t\t\t\t\t90 -45
\t\t\t\t\t\t\t140 180
"""


def test_reads_types(tmp_path):
    path = tmp_path / "om.conf"
    path.write_text(_CONF)
    types = read_om_conf(path)
    assert set(types) == {7, 8}
    assert types[7]["cable_azimuth"] == 90.0
    assert types[8]["cable_azimuth"] is None
    assert types[8]["pmt_dirs"] == [(40.0, 0.0), (90.0, -45.0), (140.0, 180.0)]


def test_round_trips_through_detector(tmp_path):
    path = tmp_path / "om.conf"
    path.write_text(_CONF)
    types = read_om_conf(path)
    modules = [Module(np.zeros(3), (1, i + 1), **kw) for i, kw in enumerate(types.values())]
    out = tmp_path / "written.conf"
    Detector(modules, Medium.ICE).to_om_conf(str(out))
    assert read_om_conf(out) == types


def test_first_definition_kept(tmp_path):
    path = tmp_path / "om.conf"
    path.write_text("A 7 1 0.4 0.1 0.1 1 180 0\nB 7 1 0.9 0.1 0.1 1 180 0\n")
    assert read_om_conf(path)[7]["beta"] == 0.4


def test_missing_directions_raise(tmp_path):
    path = tmp_path / "om.conf"
    path.write_text("A 7 1 0.4 0.1 0.1 2 180 0\nB 8 1 0.9 0.1 0.1 1 180 0\n")
    with pytest.raises(ValueError, match="declares 2 PMTs"):
        read_om_conf(path)
