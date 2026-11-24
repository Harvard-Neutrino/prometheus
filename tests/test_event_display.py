"""Tests for examples/08_event_display.py."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from prometheus.detector import Detector, Medium, Module

_EXAMPLES = Path(__file__).parents[1] / "examples"
_SPEC = importlib.util.spec_from_file_location("event_display", _EXAMPLES / "08_event_display.py")
ed = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ed)

_OM = ed.event_view.load_om(str(_EXAMPLES / "data" / "lom" / "lom16_start.conf"), 171)


def _detector():
    return Detector(
        [Module(np.array([0.0, 0.0, 20.0 * i]), (1, i + 1)) for i in range(4)], Medium.ICE
    )


def _row():
    return {
        "photons": {
            "string_id": [1, 1, 1],
            "sensor_id": [2, 2, 3],
            "t": [30.0, 10.0, 20.0],
            "pmt_id": [3, 4, 5],
        },
        "mc_truth": {
            "initial_state_type": 14,
            "initial_state_energy": 2000.0,
            "initial_state_x": -20.0,
            "initial_state_y": 0.0,
            "initial_state_z": 0.0,
            "final_state_type": [13, -2000001006],
            "final_state_energy": [1500.0, 500.0],
            "final_state_zenith": [np.pi / 2, np.pi / 2],
            "final_state_azimuth": [0.0, 0.0],
        },
    }


def _build(row):
    shared = ed.build_detector(_detector(), _OM, 15.0)
    return shared, ed.build_event(row, 7, shared["modules"], _OM["n_pmts"])


def test_build():
    shared, event = _build(_row())
    json.dumps([shared, event])
    assert len(shared["modules"]) == 4
    assert event["title"] == "Muon neutrino, 2.0 TeV"
    assert event["label"].startswith("7:")
    assert event["selected"] == 1
    assert set(event["hits"]) == {1, 2}
    assert event["hits"][1]["hits"] == [[10.0, 4], [30.0, 3]]
    np.testing.assert_allclose(event["direction"], [1, 0, 0], atol=1e-12)


def test_events_without_photons_or_muon_are_skipped():
    row = _row()
    row["photons"] = {key: [] for key in row["photons"]}
    assert _build(row)[1] is None
    row = _row()
    row["mc_truth"]["final_state_type"] = [11, -2000001006]
    assert _build(row)[1] is None


def test_track_leaves_the_detector():
    positions = np.array([[0.0, 0.0, -50.0], [0.0, 0.0, 50.0]])
    length = ed.track_length(np.array([-30.0, 0, 0]), np.array([1.0, 0, 0]), positions, margin=10.0)
    assert length == pytest.approx(90.0)


def test_direct_light_leaves_at_the_cherenkov_angle():
    angle = np.arccos(1 / ed.PHASE_INDEX)
    vertex, direction = np.zeros(3), np.array([0.6, 0.0, 0.8])
    position = np.array([100.0, 20.0, 60.0])
    point, on_cone = ed.direct_light(position, vertex, direction, 500.0, angle)
    path = position - point
    assert on_cone
    assert path @ direction / np.linalg.norm(path) == pytest.approx(np.cos(angle))
    np.testing.assert_allclose(np.cross(point, direction), 0, atol=1e-9)


@pytest.mark.parametrize("x,expected", [(-50.0, 0.0), (900.0, 500.0)])
def test_direct_light_stays_on_the_track(x, expected):
    direction = np.array([1.0, 0.0, 0.0])
    point, on_cone = ed.direct_light(np.array([x, 20.0, 0.0]), np.zeros(3), direction, 500.0, 0.7)
    assert not on_cone
    np.testing.assert_allclose(point, [expected, 0, 0])


def test_unknown_module_is_reported():
    row = _row()
    row["photons"]["sensor_id"][0] = 99
    with pytest.raises(SystemExit, match="not in the geo file"):
        _build(row)


def test_template_has_one_data_slot():
    assert ed.TEMPLATE.read_text().count("__EVENT_DATA__") == 1
