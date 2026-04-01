# -*- coding: utf-8 -*-
"""
Tests for configuration system.
"""
import tempfile

import numpy as np
import pytest

from fennel import config
from fennel.config import ConfigClass


@pytest.mark.unit
class TestConfig:
    """Test suite for configuration system."""

    def test_config_is_dict(self):
        """Test that config behaves as a dictionary."""
        assert isinstance(config, dict)
        assert "general" in config
        assert "scenario" in config

    def test_config_default_values(self):
        """Test that default config values are set correctly."""
        assert "random state seed" in config["general"]
        assert "jax" in config["general"]
        assert "medium" in config["scenario"]
        assert config["scenario"]["medium"] in ["water", "ice"]

    def test_config_mediums(self):
        """Test that medium parameters are defined."""
        for medium in ["water", "ice"]:
            assert medium in config["mediums"]
            assert "refractive index" in config["mediums"][medium]
            assert "density" in config["mediums"][medium]
            assert "radiation length" in config["mediums"][medium]

    def test_config_particles(self):
        """Test that particle configurations are defined."""
        assert "track particles" in config["simulation"]
        assert "em particles" in config["simulation"]
        assert "hadron particles" in config["simulation"]

        assert 13 in config["simulation"]["track particles"]
        assert 11 in config["simulation"]["em particles"]
        assert 211 in config["simulation"]["hadron particles"]

    def test_config_pdg_ids(self):
        """Test that PDG ID mapping is defined."""
        assert "pdg id" in config
        assert 11 in config["pdg id"]
        assert config["pdg id"][11] == "e-"
        assert config["pdg id"][13] == "mu-"

    def test_config_from_dict(self):
        """Test updating config from dictionary."""
        test_config = ConfigClass({})
        user_dict = {"general": {"random state seed": 123, "jax": True}}
        test_config.from_dict(user_dict)

        assert test_config["general"]["random state seed"] == 123
        assert test_config["general"]["jax"] is True

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        test_config = ConfigClass({})

        yaml_content = """
general:
  random state seed: 456
  jax: false
scenario:
  medium: ice
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name

        try:
            test_config.from_yaml(yaml_file)
            assert test_config["general"]["random state seed"] == 456
            assert test_config["general"]["jax"] is False
            assert test_config["scenario"]["medium"] == "ice"
        finally:
            import os

            os.unlink(yaml_file)

    def test_config_advanced_settings(self):
        """Test that advanced settings are defined."""
        assert "advanced" in config
        assert "fine structure" in config["advanced"]
        assert "wavelengths" in config["advanced"]
        assert "angles" in config["advanced"]
        assert "z grid" in config["advanced"]

    def test_wavelength_range(self):
        """Test that wavelength range is reasonable."""
        wavelengths = config["advanced"]["wavelengths"]
        assert isinstance(wavelengths, np.ndarray)
        assert len(wavelengths) > 0
        assert np.all(wavelengths > 0)
        assert np.all(wavelengths < 1000)  # Reasonable range for optical

    def test_angle_range(self):
        """Test that angle range is 0-180 degrees."""
        angles = config["advanced"]["angles"]
        assert isinstance(angles, np.ndarray)
        assert len(angles) > 0
        assert np.min(angles) >= 0
        assert np.max(angles) <= 180

    def test_track_interactions(self):
        """Test that track interactions are defined."""
        interactions = config["simulation"]["track interactions"]
        expected = ["ionization", "pair", "brems", "nuclear", "total"]
        for interaction in expected:
            assert interaction in interactions

    def test_em_cascade_parameters(self):
        """Test that EM cascade parameters exist for all particles."""
        em_particles = config["simulation"]["em particles"]

        for particle in em_particles:
            assert particle in config["em cascade"]["track parameters"]
            assert particle in config["em cascade"]["longitudinal parameters"]
            assert particle in config["em cascade"]["angular distribution"]

    def test_hadron_cascade_parameters(self):
        """Test that hadron cascade parameters exist for all particles."""
        hadron_particles = config["simulation"]["hadron particles"]

        for particle in hadron_particles:
            assert particle in config["hadron cascade"]["track parameters"]
            assert particle in config["hadron cascade"]["em fraction"]
            assert particle in config["hadron cascade"]["longitudinal parameters"]
            assert particle in config["hadron cascade"]["angular distribution"]

    def test_track_parameters_structure(self):
        """Test structure of track parameters."""
        track_params = config["track"]

        assert "additional track water" in track_params
        assert "additional track ice" in track_params
        assert "angular distribution" in track_params

        for interaction in ["ionization", "pair", "brems", "nuclear", "total"]:
            assert interaction in track_params["additional track water"]
            params = track_params["additional track water"][interaction]
            assert "lambda" in params
            assert "kappa" in params

    def test_config_modification(self):
        """Test that config can be modified."""
        original_seed = config["general"]["random state seed"]

        # Modify
        config["general"]["random state seed"] = 999
        assert config["general"]["random state seed"] == 999

        # Restore
        config["general"]["random state seed"] = original_seed

    def test_refractive_indices(self):
        """Test that refractive indices are physical."""
        for medium, params in config["mediums"].items():
            n = params["refractive index"]
            assert n >= 1.0  # Physical constraint
            assert n < 2.0  # Reasonable for water/ice

    def test_density_values(self):
        """Test that density values are physical."""
        for medium, params in config["mediums"].items():
            density = params["density"]
            assert density > 0
            assert density < 2.0  # Reasonable for water/ice
