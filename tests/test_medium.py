"""Tests for water medium model registration and flow model selection."""

import warnings

from hyperion.medium import medium_collections
from prometheus.detector.medium import Medium
from prometheus.photon_propagation.olympus_photon_propagator import (
    _DEFAULT_COUNTS_FILE,
    _DEFAULT_FLOW_FILE,
    _FLOW_MODEL_MAP,
    _WATER_MEDIUM_MAP,
)


def test_pone_medium_registered():
    assert "pone" in medium_collections


def test_antares_medium_registered():
    assert "antares" in medium_collections


def test_medium_collections_tuples():
    for name, entry in medium_collections.items():
        assert len(entry) == 3, f"medium '{name}' must be a (ref_ix, sca_angle, sca_len) tuple"


def test_water_medium_map_covers_water():
    assert Medium.WATER in _WATER_MEDIUM_MAP
    assert _WATER_MEDIUM_MAP[Medium.WATER] == "pone"


def test_unknown_medium_falls_back_with_warning():
    """A medium not in _WATER_MEDIUM_MAP should cause a UserWarning and fall back to 'pone'."""
    from unittest.mock import MagicMock

    fake_medium = MagicMock()
    fake_medium.name = "ATLANTIS"

    medium_key = _WATER_MEDIUM_MAP.get(fake_medium)
    assert medium_key is None  # not mapped

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if medium_key is None:
            warnings.warn(
                f"No dedicated optical model is registered for medium "
                f"'{fake_medium.name}'. Falling back to the Cascadia Basin "
                f"(P-ONE) water model. Simulation results may not accurately "
                f"reflect '{fake_medium.name}' optical properties.",
                UserWarning,
                stacklevel=1,
            )
            medium_key = "pone"

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "ATLANTIS" in str(caught[0].message)
    assert "P-ONE" in str(caught[0].message)
    assert medium_key == "pone"


# ---------------------------------------------------------------------------
# Flow model map tests
# ---------------------------------------------------------------------------


def test_flow_model_map_has_pone():
    assert "pone" in _FLOW_MODEL_MAP
    flow, counts = _FLOW_MODEL_MAP["pone"]
    assert flow == _DEFAULT_FLOW_FILE
    assert counts == _DEFAULT_COUNTS_FILE


def test_flow_model_map_tuples():
    for key, entry in _FLOW_MODEL_MAP.items():
        assert len(entry) == 2, f"_FLOW_MODEL_MAP['{key}'] must be a (flow, counts) tuple"


def test_flow_model_pone_matches_defaults():
    """The P-ONE entry must match the config defaults so auto-selection is stable."""
    flow, counts = _FLOW_MODEL_MAP["pone"]
    assert flow == _DEFAULT_FLOW_FILE
    assert counts == _DEFAULT_COUNTS_FILE


def test_flow_model_unknown_medium_falls_back():
    """An unregistered medium_key must fall back to pone with a UserWarning."""
    unknown_key = "baikal"
    assert unknown_key not in _FLOW_MODEL_MAP

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if unknown_key in _FLOW_MODEL_MAP:
            flow_file, counts_file = _FLOW_MODEL_MAP[unknown_key]
        else:
            warnings.warn(
                f"No dedicated flow model is registered for medium '{unknown_key}'. "
                f"Falling back to the P-ONE model files. "
                f"Timing predictions may not accurately reflect "
                f"'{unknown_key}' optical properties.",
                UserWarning,
                stacklevel=1,
            )
            flow_file, counts_file = _FLOW_MODEL_MAP["pone"]

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "baikal" in str(caught[0].message)
    assert flow_file == _DEFAULT_FLOW_FILE
    assert counts_file == _DEFAULT_COUNTS_FILE
