"""Regression tests for LeptonInjector cross-section spline selection.

Guards against the nu/nubar swap in ``_li_injection_config_mims``: the spline
must match the primary neutrino LeptonInjector derives from ``final_state_1``
(see LI ``deduceInitialType``). A "Minus"/plain final state (MuMinus, TauMinus,
NuMu, ...) comes from a matter neutrino -> ``nu``; a "Plus"/"Bar" one (MuPlus,
TauPlus, NuTauBar, ...) from an antineutrino -> ``nubar``.
"""

from types import SimpleNamespace

import pytest

from prometheus.config_types import LeptonInjectorConfig
from prometheus.utils.config_mims import _li_injection_config_mims


@pytest.fixture
def _no_geo(monkeypatch):
    """Stub the geometry helper so the test needs no real detector geometry."""
    import prometheus.utils.geo_utils as geo_utils

    monkeypatch.setattr(geo_utils, "get_volume", lambda coords, is_ice: (0.0, 0.0))


def _run(final_state_1: str, final_state_2: str = "Hadrons"):
    """Fill in a minimal LI config and return the computed spline paths."""
    config = LeptonInjectorConfig()
    config.simulation.final_state_1 = final_state_1
    config.simulation.final_state_2 = final_state_2
    # Pre-fill the geometry-derived fields so their get_* helpers are skipped;
    # only the xsec-path logic under test is exercised.
    config.simulation.is_ranged = False
    config.simulation.endcap_length = 1.0
    config.simulation.injection_radius = 1.0
    config.simulation.cylinder_radius = 1.0
    config.simulation.cylinder_height = 1.0

    detector = SimpleNamespace(
        medium=SimpleNamespace(name="ICE"),
        module_coords=None,
    )
    _li_injection_config_mims(
        config,
        detector,
        nevents=1,
        seed=1,
        output_prefix="test",
        earth_model_file="PREM_south_pole.dat",
    )
    return config.paths.diff_xsec, config.paths.total_xsec


@pytest.mark.parametrize(
    "final_state_1, final_state_2, expected_nutype, expected_int",
    [
        # CC: charged lepton final state
        ("TauMinus", "Hadrons", "nu", "CC"),
        ("TauPlus", "Hadrons", "nubar", "CC"),
        ("MuMinus", "Hadrons", "nu", "CC"),
        ("MuPlus", "Hadrons", "nubar", "CC"),
        ("EMinus", "Hadrons", "nu", "CC"),
        ("EPlus", "Hadrons", "nubar", "CC"),
        # NC: outgoing neutrino final state ("Bar" -> antineutrino)
        ("NuTau", "Hadrons", "nu", "NC"),
        ("NuTauBar", "Hadrons", "nubar", "NC"),
        ("NuMu", "Hadrons", "nu", "NC"),
        ("NuMuBar", "Hadrons", "nubar", "NC"),
    ],
)
def test_xsec_nutype_matches_primary(
    _no_geo, final_state_1, final_state_2, expected_nutype, expected_int
):
    diff_xsec, total_xsec = _run(final_state_1, final_state_2)
    assert diff_xsec.endswith(f"dsdxdy_{expected_nutype}_{expected_int}_iso.fits")
    assert total_xsec.endswith(f"sigma_{expected_nutype}_{expected_int}_iso.fits")
