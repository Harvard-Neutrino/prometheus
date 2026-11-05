"""Tests for the pooling of counts labels before the counts-model fit."""

import numpy as np
import pytest

from hyperion.models.photon_arrival_time_nflow.training_data import pool_counts_rows


def _grid(label):
    """Build a counts table on a distance x angle grid from a label function."""
    dist = np.geomspace(1.0, 300.0, 40)
    angle = np.radians(np.arange(5.0, 180.0, 10.0))
    d, a = (x.ravel() for x in np.meshgrid(dist, angle, indexing="ij"))
    return {
        "log10_dist": np.log10(d),
        "angle": a,
        "log10_survival": np.log10(label(d, a)),
        "n_detected": np.full(d.size, 1000.0),
        "n_effective": np.full(d.size, 1000.0),
    }


def _smooth(d, a):
    return np.exp(-d / 40.0) / d**2 * np.ones_like(a)


def test_near_rows_are_untouched():
    rows = _grid(_smooth)
    pooled = pool_counts_rows(rows)
    near = rows["log10_dist"] < np.log10(8.0)
    assert np.allclose(pooled["log10_survival"][near], rows["log10_survival"][near])


def test_smooth_table_is_preserved():
    rows = _grid(_smooth)
    pooled = pool_counts_rows(rows)
    assert np.max(np.abs(pooled["log10_survival"] - rows["log10_survival"])) < 0.05


def test_heavy_row_is_shared_and_light_is_kept():
    rows = _grid(_smooth)
    far = np.flatnonzero(
        np.isclose(rows["log10_dist"], np.unique(rows["log10_dist"])[-8])
        & np.isclose(rows["angle"], np.radians(95.0))
    )
    rows["log10_survival"][far] += 1.0
    rows["n_effective"][far] = 1.0
    pooled = pool_counts_rows(rows)
    assert pooled["log10_survival"][far] < rows["log10_survival"][far] - 0.5
    weight = np.sin(rows["angle"])
    raw = np.sum(weight * 10.0 ** rows["log10_survival"] * 10.0 ** (2 * rows["log10_dist"]))
    new = np.sum(weight * 10.0 ** pooled["log10_survival"] * 10.0 ** (2 * rows["log10_dist"]))
    assert new == pytest.approx(raw, rel=0.15)


def test_grid_must_be_rectangular():
    rows = _grid(_smooth)
    with pytest.raises(ValueError):
        pool_counts_rows({key: value[:-1] for key, value in rows.items()})
