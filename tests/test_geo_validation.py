"""Geometry validation: a bad geo file must fail loudly and say why.

Every problem is reported in one ``GeometryError`` so the file can be fixed
in a single pass. Duplicate ``(string_id, om_id)`` keys are the important
case: hits are matched back to modules by key, so a duplicate silently sends
light to the wrong module.
"""

import warnings

import numpy as np
import pytest

from prometheus.detector import Detector, GeometryError, Medium, Module, detector_from_geo

HEADER = "### Metadata ###\nMedium:\twater\n### Modules ###\n"


def _write_geo(tmp_path, rows, header=HEADER):
    path = tmp_path / "test.geo"
    body = "".join("\t".join(str(v) for v in row) + "\n" for row in rows)
    path.write_text(header + body)
    return str(path)


def _string(sid, x, y, n=3, om0=0):
    return [(x, y, 10.0 * i, sid, om0 + i) for i in range(n)]


def test_good_file_loads_without_warnings(tmp_path):
    path = _write_geo(tmp_path, _string(0, 0, 0) + _string(1, 50, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        det = detector_from_geo(path)
    assert det.n_modules == 6


def test_survey_drift_within_tolerance_is_fine(tmp_path):
    rows = [(0.0, 0.0, 0.0, 0, 0), (0.6, -0.4, 10.0, 0, 1)]
    path = _write_geo(tmp_path, rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detector_from_geo(path)


def test_reused_string_id_reports_both_positions(tmp_path):
    # Two physical strings at different (x, y) both labelled string 3.
    path = _write_geo(tmp_path, _string(3, 0, 0) + _string(3, 50, 20))
    with pytest.raises(GeometryError) as excinfo:
        detector_from_geo(path)
    msg = str(excinfo.value)
    assert "test.geo" in msg
    assert "3 duplicate (string_id, om_id) key(s)" in msg
    assert "(3, 0)" in msg
    assert "string 3 at (0, 0); (50, 20)" in msg


def test_all_problems_reported_together(tmp_path):
    rows = _string(0, 0, 0)
    rows.append((1.0, 2.0))  # too few fields
    rows.append(("a", 0.0, 0.0, 1, 0))  # non-numeric position
    rows.append((0.0, 0.0, "nan", 1, 1))  # non-finite position
    rows.append((0.0, 0.0, 0.0, 1, "x"))  # non-integer id
    path = _write_geo(tmp_path, rows)
    with pytest.raises(GeometryError) as excinfo:
        detector_from_geo(path)
    problems = excinfo.value.problems
    assert len(problems) == 4
    assert problems[0].startswith("line 7:") and "expected 5" in problems[0]
    assert problems[1].startswith("line 8:") and "not numeric" in problems[1]
    assert problems[2].startswith("line 9:") and "not finite" in problems[2]
    assert problems[3].startswith("line 10:") and "not integers" in problems[3]


def test_missing_modules_header(tmp_path):
    path = _write_geo(tmp_path, _string(0, 0, 0), header="### Metadata ###\nMedium:\twater\n")
    with pytest.raises(GeometryError, match="missing '### Modules ###'"):
        detector_from_geo(path)


def test_no_modules(tmp_path):
    path = _write_geo(tmp_path, [])
    with pytest.raises(GeometryError, match="no modules listed"):
        detector_from_geo(path)


def test_blank_lines_are_ignored(tmp_path):
    path = _write_geo(tmp_path, _string(0, 0, 0))
    with open(path, "a") as f:
        f.write("\n\n")
    assert detector_from_geo(path).n_modules == 3


def test_negative_ids_warn_but_load(tmp_path):
    path = _write_geo(tmp_path, _string(-1, 0, 0) + _string(1, 50, 0))
    with pytest.warns(UserWarning, match="negative string or OM ID"):
        det = detector_from_geo(path)
    assert det.n_modules == 6


def test_unaligned_string_warns(tmp_path):
    rows = [(0.0, 0.0, 0.0, 0, 0), (30.0, 0.0, 10.0, 0, 1)]
    path = _write_geo(tmp_path, rows)
    with pytest.warns(UserWarning, match="not on one vertical line"):
        detector_from_geo(path)


class TestDetectorConstructor:
    def test_duplicate_keys_rejected(self):
        mods = [Module(np.array([0.0, 0.0, z]), (0, 0)) for z in (0.0, 10.0)]
        with pytest.raises(GeometryError, match="duplicate"):
            Detector(mods, Medium.WATER)

    def test_add_with_overlapping_keys_rejected(self):
        a = Detector([Module(np.array([0.0, 0.0, 0.0]), (0, 0))], Medium.WATER)
        b = Detector([Module(np.array([50.0, 0.0, 0.0]), (0, 0))], Medium.WATER)
        with pytest.raises(GeometryError, match="same ID"):
            a + b

    def test_non_finite_position_rejected(self):
        mods = [Module(np.array([0.0, np.inf, 0.0]), (0, 0))]
        with pytest.raises(GeometryError, match="non-finite"):
            Detector(mods, Medium.WATER)

    def test_empty_rejected(self):
        with pytest.raises(GeometryError, match="no modules"):
            Detector([], Medium.WATER)
