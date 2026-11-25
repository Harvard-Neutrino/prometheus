"""Tests for the multi-PMT part of examples/04_event_view.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES = Path(__file__).parents[1] / "examples"
_SPEC = importlib.util.spec_from_file_location("event_view", _EXAMPLES / "04_event_view.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

_OM = ev.load_om(str(_EXAMPLES / "data" / "lom" / "lom16_start.conf"), 171)
_CYLINDER = dict(_OM, Rr=0.06, Rz=-0.4)


def _photons(n=60):
    rng = np.random.default_rng(1)
    return {
        "sensor_pos_x": np.zeros(n),
        "sensor_pos_y": np.zeros(n),
        "sensor_pos_z": np.where(np.arange(n) < 40, 0.0, 17.0),
        "string_id": np.ones(n, dtype=int),
        "sensor_id": np.where(np.arange(n) < 40, 1, 2),
        "t": rng.uniform(0, 100, n),
        "pmt_id": rng.integers(0, 16, n),
        "photon_zenith": np.full(n, np.pi / 2),
        "photon_azimuth": np.full(n, np.pi),
    }


@pytest.mark.parametrize("om", [_OM, _CYLINDER])
@pytest.mark.parametrize("shape", ["vessel", "ppc"])
def test_surface_distance_at_axes(om, shape):
    directions = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    expected = [om["Rr"], om["Rr"], abs(om["Rz"]), abs(om["Rz"])]
    np.testing.assert_allclose(ev.surface_distance(om, shape, directions), expected)


@pytest.mark.parametrize("om", [_OM, _CYLINDER])
@pytest.mark.parametrize("shape", ["vessel", "ppc"])
def test_profile_lies_on_surface(om, shape):
    radius, height = ev.profile(om, shape)
    length = np.hypot(radius, height)
    directions = np.stack([radius, np.zeros_like(radius), height], axis=-1) / length[:, None]
    np.testing.assert_allclose(ev.surface_distance(om, shape, directions), length, rtol=1e-6)


def test_selects_brightest_module_and_counts():
    photons = _photons()
    key, mask = ev.select_module(photons, None)
    assert key == (1, 1)
    counts = ev.pmt_counts(photons, mask, 16)
    assert counts.sum() == 40
    np.testing.assert_allclose(
        ev.mean_photon_direction(photons, mask), [-1.0, 0.0, 0.0], atol=1e-12
    )


def test_bad_pmt_ids_are_reported():
    photons = _photons()
    mask = np.ones(60, dtype=bool)
    with pytest.raises(SystemExit, match="does not fit"):
        ev.pmt_counts(photons, mask, 4)
    photons["pmt_id"] = np.full(60, np.nan)
    with pytest.raises(SystemExit, match="no PMT"):
        ev.pmt_counts(photons, mask, 16)


def test_first_view_is_on_the_light_side():
    travel = np.array([-1.0, 0.0, 0.0])
    facing, far = ev.side_views(travel)
    assert facing @ travel < 0 < far @ travel
    assert np.linalg.norm(facing) == pytest.approx(1.0)


@pytest.mark.parametrize("om", [None, _OM])
def test_draws(om):
    import matplotlib

    matplotlib.use("Agg")
    photons = _photons()
    args = ev.parse_args(["unused.parquet"])
    hits = ev.aggregate_hits(photons)
    truth = {"initial_state_energy": 1.0, "initial_state_type": 211}
    fig = ev.draw_event(hits, None, "ice", truth, 0, args, photons, om)
    fig.canvas.draw()
    assert len(fig.axes) == (2 if om is None else 5)
    ev.plt.close(fig)
