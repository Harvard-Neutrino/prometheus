"""Module types in geo files: the optional sixth field and the OM conf line."""

import numpy as np
import pytest

from prometheus.detector import Detector, GeometryError, Medium, Module, detector_from_geo
from prometheus.utils.geo_utils import from_geo

OM_CONF = "a\t5\t1\t0.9\t0.15\t0.2\t2\t45 0\n\t\t\t\t\t\t\t135 90\n"
TWO_PMTS = dict(n_pmts=2, pmt_dirs=[(45.0, 0.0), (135.0, 90.0)])


def _write(tmp_path, types, om_conf="types/om.conf", text=OM_CONF):
    (tmp_path / "types").mkdir(exist_ok=True)
    (tmp_path / "types" / "om.conf").write_text(text)
    lines = ["### Metadata ###"]
    if om_conf is not None:
        lines.append(f"OM conf:\t{om_conf}")
    lines += ["Medium:\tice", "### Modules ###"]
    for i, t in enumerate(types):
        line = f"0.0\t0.0\t{10.0 * i}\t1\t{i}"
        lines.append(line if t is None else f"{line}\t{t}")
    path = tmp_path / "test.geo"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_types_take_their_shape_from_om_conf(tmp_path):
    det = detector_from_geo(_write(tmp_path, [5, None, -1]))
    first, second, third = det.modules
    assert first.module_type == 5
    assert (first.Rr, first.Rz, first.beta) == (0.15, 0.2, 0.9)
    assert first.pmt_dirs == TWO_PMTS["pmt_dirs"]
    assert second.module_type == third.module_type == -1
    assert second.n_pmts == third.n_pmts == 1
    assert det.needs_nextgen()


def test_efficiency_and_noise_still_apply(tmp_path):
    path = _write(tmp_path, [5, None])
    det = detector_from_geo(path, efficiency=0.4, noise_rate=7)
    assert [(m.efficiency, m.noise_rate) for m in det.modules] == [(0.4, 7), (0.4, 7)]


def test_relative_path_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    path = _write(tmp_path, [5])
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    assert detector_from_geo(path).modules[0].module_type == 5


def test_absolute_path(tmp_path):
    path = _write(tmp_path, [5], om_conf=tmp_path / "types" / "om.conf")
    assert detector_from_geo(path).modules[0].module_type == 5


def test_path_with_the_word_medium(tmp_path):
    (tmp_path / "medium").mkdir()
    (tmp_path / "medium" / "om.conf").write_text(OM_CONF)
    det = detector_from_geo(_write(tmp_path, [5], om_conf="medium/om.conf"))
    assert det.medium is Medium.ICE


@pytest.mark.parametrize(
    "types, kwargs, message",
    [
        ([5], {"om_conf": None}, "line 4: type_id 5 needs an 'OM conf:' line"),
        ([5, 6], {}, "line 6: type_id 6 is not defined"),
        ([-2], {}, "type_id -2 is not defined"),
        (["x"], {}, "line 5: type_id is not an integer"),
        ([5], {"om_conf": "missing.conf"}, "cannot read .*missing.conf"),
        ([5], {"om_conf": ""}, "has no path"),
        ([5], {"text": "a 5 1\n"}, "cannot read .*Malformed"),
        ([None], {}, "no module has a type_id"),
        (["5\t5"], {}, "expected 5 or 6 tab-separated fields .* got 7"),
    ],
)
def test_bad_files_are_rejected(tmp_path, types, kwargs, message):
    with pytest.raises(GeometryError, match=message):
        detector_from_geo(_write(tmp_path, types, **kwargs))


def _mixed_detector(**changes):
    shape = dict(module_type=5, Rr=0.15, Rz=0.2, beta=0.9, area=1.1, cable_azimuth=30.0, **TWO_PMTS)
    return Detector(
        [
            Module(np.array([0.0, 0.0, 0.0]), (1, 1), **shape),
            Module(np.array([0.0, 0.0, 10.0]), (1, 2), **{**shape, **changes}),
            Module(np.array([0.0, 0.0, 20.0]), (1, 3)),
        ],
        Medium.ICE,
    )


def test_round_trip(tmp_path):
    det = _mixed_detector()
    path = str(tmp_path / "out.geo")
    det.to_geo(path)
    again = detector_from_geo(path)
    for a, b in zip(det.modules, again.modules):
        for name in ("key", "module_type", "Rr", "Rz", "beta", "area", "n_pmts", "cable_azimuth"):
            assert getattr(a, name) == getattr(b, name)
        assert a.pmt_dirs == b.pmt_dirs
        np.testing.assert_allclose(a.pos, b.pos)


def test_round_trip_of_a_moved_pair(tmp_path):
    _mixed_detector().to_geo(str(tmp_path / "out.geo"))
    moved = tmp_path / "moved"
    moved.mkdir()
    for name in ("out.geo", "out.om.conf"):
        (tmp_path / name).rename(moved / name)
    assert detector_from_geo(str(moved / "out.geo")).modules[0].module_type == 5


@pytest.mark.parametrize("changes", [{"Rr": 0.16}, {"cable_azimuth": None}])
def test_types_that_differ_are_not_written(tmp_path, changes):
    path = tmp_path / "out.geo"
    with pytest.raises(ValueError, match="type 5 differ"):
        _mixed_detector(**changes).to_geo(str(path))
    assert not path.exists()


def test_legacy_detector_writes_the_old_format(tmp_path):
    det = Detector([Module(np.array([0.0, 0.0, 1.0]), (1, 2))], Medium.ICE)
    path = tmp_path / "out.geo"
    det.to_geo(str(path))
    assert path.read_text() == (
        "### Metadata ###\nMedium:\tice\n### Modules ###\n0.0\t0.0\t1.0\t1\t2\n"
    )
    assert list(tmp_path.iterdir()) == [path]


def test_from_geo_ignores_the_type(tmp_path):
    pos, keys, medium = from_geo(_write(tmp_path, [5, None]))
    assert keys == [(1, 0), (1, 1)]
    assert medium == "ice"
    assert pos.shape == (2, 3)
