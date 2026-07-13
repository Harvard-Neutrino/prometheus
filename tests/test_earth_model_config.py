"""Earth-model resolution in ``config_mims``: the explicit
``config.detector.earth_model`` option (site key or absolute path) and the
legacy geo-name lookup with its now-warned medium fallback."""

import importlib
import logging
import tempfile
from types import SimpleNamespace

import pytest

from prometheus.detector.medium import Medium

# the package re-exports the config_mims *function*, shadowing the submodule
config_mims_module = importlib.import_module("prometheus.utils.config_mims")

DENSITIES = config_mims_module.RESOURCES_DIR / "earthparams" / "densities"
DETECTOR = SimpleNamespace(medium=Medium.WATER, _offset=[0.0, 0.0, -3000.0])


class _Indexable(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


def _make_config(geo_file, storage_prefix, earth_model=None):
    return SimpleNamespace(
        photon_propagator=SimpleNamespace(name="PPC_CUDA"),
        run=SimpleNamespace(
            random_state_seed=1,
            run_number=1,
            storage_prefix=storage_prefix,
            outfile="out.parquet",
            nevents=1,
        ),
        detector=SimpleNamespace(geo_file=geo_file, offset=None, earth_model=earth_model),
        injection=_Indexable(name="ranged", ranged=SimpleNamespace()),
        lepton_propagator=_Indexable(name="proposal", proposal=SimpleNamespace()),
    )


def _resolved_earth_model(monkeypatch, geo_file, earth_model=None):
    """Run ``config_mims`` with stubbed downstream helpers; return the model."""
    seen = {}
    for helper in ("lepton_prop_config_mims", "photon_prop_config_mims", "check_consistency"):
        monkeypatch.setattr(config_mims_module, helper, lambda *a, **k: None)
    monkeypatch.setattr(
        config_mims_module, "injection_config_mims", lambda *args: seen.update(file=args[-1])
    )
    with tempfile.TemporaryDirectory() as tmp:
        config_mims_module.config_mims(_make_config(geo_file, tmp, earth_model), DETECTOR)
    return seen["file"]


def test_site_key(monkeypatch):
    resolved = _resolved_earth_model(monkeypatch, "my_custom.geo", earth_model="arca")
    assert resolved == str(DENSITIES / "PREM_arca.dat")


def test_absolute_path(monkeypatch, tmp_path):
    model = tmp_path / "my_model.dat"
    model.write_text("# fake earth model\n")
    resolved = _resolved_earth_model(monkeypatch, "whatever.geo", earth_model=str(model))
    assert resolved == str(model)


def test_unknown_model_raises_with_available_keys(monkeypatch):
    with pytest.raises(ValueError, match="atlantis.*PREM_arca"):
        _resolved_earth_model(monkeypatch, "whatever.geo", earth_model="atlantis")


def test_known_geo_name_unchanged(monkeypatch):
    resolved = _resolved_earth_model(monkeypatch, "arca.geo")
    assert resolved == str(DENSITIES / "PREM_arca.dat")


def test_medium_fallback_warns(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="prometheus.utils.config_mims"):
        resolved = _resolved_earth_model(monkeypatch, "arca_displaced.geo")
    assert resolved == str(DENSITIES / "PREM_water.dat")
    assert any("earth_model" in r.message for r in caplog.records)


def test_lepton_prop_consumer_accepts_absolute_path(tmp_path):
    model = tmp_path / "m.dat"
    model.write_text("# fake\n")
    cfg = SimpleNamespace(
        simulation=SimpleNamespace(medium=None, propagation_padding=100.0),
        paths=SimpleNamespace(earth_model_location=None),
    )
    config_mims_module.lepton_prop_config_mims(cfg, DETECTOR, str(model))
    assert cfg.paths.earth_model_location == str(model)
