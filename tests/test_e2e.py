"""End-to-end regression tests.

Each test runs a full simulation with a fixed RNG seed and asserts that the
physics output (total photon hit count) matches a reference value captured on
the baseline commit. A tolerance of ±1 % is applied to guard against any
platform-level floating-point differences while still catching real regressions.

These tests are marked ``slow`` and are excluded from the default pytest run.
Run them explicitly with::

    pytest -m slow

or to run all tests including slow ones::

    pytest --run-slow
"""
import copy
import glob
import os

import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Reference values (captured with seed=42, 100 events, baseline commit)
# ---------------------------------------------------------------------------
WATER_REF_HITS = 25193   # total photon arrivals across 100 events
WATER_TOL = 0.01         # allow ±1 % for platform floating-point differences


def _count_hits(parquet_path: str) -> int:
    """Return total photon hit count across all events in a parquet file."""
    tbl = pq.read_table(parquet_path)
    photons = tbl.column("photons").to_pylist()
    return sum(len(row["t"]) for row in photons)


@pytest.mark.slow
def test_e2e_water(tmp_path):
    """100-event water simulation (olympus/JAX) reproduces reference hit count."""
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    from prometheus import Prometheus, config as _config

    cfg = copy.deepcopy(_config)
    cfg["run"]["run number"] = 9901
    cfg["run"]["random state seed"] = 42
    cfg["run"]["nevents"] = 100
    cfg["run"]["storage prefix"] = str(tmp_path) + "/"
    cfg["injection"]["name"] = "LeptonInjector"
    cfg["injection"]["LeptonInjector"]["simulation"]["is ranged"] = False
    cfg["injection"]["LeptonInjector"]["simulation"]["minimal energy"] = 1e3
    cfg["injection"]["LeptonInjector"]["simulation"]["maximal energy"] = 1e4
    cfg["detector"]["geo file"] = "resources/geofiles/demo_water.geo"

    prom = Prometheus(cfg)
    prom.sim()

    files = glob.glob(str(tmp_path / "9901*.parquet"))
    assert len(files) == 1, f"Expected 1 output file, found: {files}"

    hits = _count_hits(files[0])
    assert abs(hits - WATER_REF_HITS) / WATER_REF_HITS <= WATER_TOL, (
        f"Photon hit count {hits} deviates more than {WATER_TOL*100:.0f}% "
        f"from reference {WATER_REF_HITS}"
    )
