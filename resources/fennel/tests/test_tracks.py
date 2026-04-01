# -*- coding: utf-8 -*-
"""
Unit tests for Track class.

Tests the track light yield calculations for muons.
"""
import numpy as np
import pytest

from fennel import config
from fennel.tracks import Track
from tests.conftest import trapezoid_compat


@pytest.mark.unit
class TestTrack:
    """Test suite for Track class."""

    def test_track_initialization(self):
        """Test Track object can be initialized."""
        config["general"]["enable logging"] = False
        track = Track()
        assert track is not None
        assert hasattr(track, "additional_track_ratio")
        assert hasattr(track, "cherenkov_angle_distro")

    def test_additional_track_ratio_shape(self):
        """Test that additional track ratio returns correct shape."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        track = Track()

        energies = np.array([1.0, 10.0, 100.0])
        interactions = ["ionization", "pair", "brems", "nuclear", "total"]

        for interaction in interactions:
            ratio = track.additional_track_ratio(energies, interaction)
            assert ratio.shape == energies.shape
            assert np.all(np.isfinite(ratio))

    def test_additional_track_ratio_single_energy(self):
        """Test additional track ratio with single energy value."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        track = Track()

        energy = 100.0
        ratio = track.additional_track_ratio(energy, "total")
        assert isinstance(ratio, (float, np.ndarray))
        assert np.isfinite(ratio)
        assert ratio >= 0

    def test_cherenkov_angle_distribution(self):
        """Test Cherenkov angle distribution."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        track = Track()

        angles = np.linspace(0.0, 180.0, 50)
        n = 1.333  # water
        energy = 100.0

        angle_dist = track.cherenkov_angle_distro(angles, n, energy)
        assert angle_dist.shape == angles.shape
        assert np.all(np.isfinite(angle_dist))
        assert np.all(angle_dist >= 0)

        # Check normalization (should integrate to something reasonable)
        integral = trapezoid_compat(angle_dist, angles)
        assert integral > 0

    def test_track_ratio_energy_dependence(self):
        """Test that track ratio varies with energy as expected."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        track = Track()

        low_energy = 1.0
        high_energy = 10000.0

        ratio_low = track.additional_track_ratio(low_energy, "total")
        ratio_high = track.additional_track_ratio(high_energy, "total")

        # Values should be different (physics dependence on energy)
        assert ratio_low != ratio_high

    def test_interaction_types_valid(self):
        """Test that all interaction types work correctly."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        track = Track()

        energy = 100.0
        interactions = ["ionization", "pair", "brems", "nuclear", "total"]

        for interaction in interactions:
            try:
                ratio = track.additional_track_ratio(energy, interaction)
                assert np.isfinite(ratio)
            except Exception as e:
                pytest.fail(f"Interaction '{interaction}' failed: {e}")

    def test_angle_distribution_symmetry(self):
        """Test angle distribution properties."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        track = Track()

        angles = np.linspace(0.0, 180.0, 100)
        n = 1.333
        energy = 100.0

        dist = track.cherenkov_angle_distro(angles, n, energy)

        # Check that distribution peaks somewhere (Cherenkov angle)
        max_idx = np.argmax(dist)
        assert 0 < max_idx < len(angles) - 1  # Peak not at boundaries

        # Cherenkov angle should be around arccos(1/n)
        expected_angle_rad = np.arccos(1.0 / n)
        expected_angle_deg = np.degrees(expected_angle_rad)

        # Peak should be reasonably close to expected Cherenkov angle
        peak_angle = angles[max_idx]
        assert abs(peak_angle - expected_angle_deg) < 10.0  # Within 10 degrees

    def test_vectorized_operation(self):
        """Test that vectorized operations work correctly."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        track = Track()

        energies = np.logspace(0, 4, 20)

        # Should handle array input
        ratios = track.additional_track_ratio(energies, "total")
        assert ratios.shape == energies.shape
        assert np.all(np.isfinite(ratios))


@pytest.mark.unit
@pytest.mark.jax
class TestTrackJAX:
    """Test Track class with JAX backend."""

    def test_track_jax_initialization(self):
        """Test Track works with JAX enabled."""
        pytest.importorskip("jax")
        config["general"]["enable logging"] = False
        config["general"]["jax"] = True
        config["general"]["random state seed"] = 42

        track = Track()
        assert track is not None

    def test_track_jax_basic_calculation(self):
        """Test basic calculation with JAX."""
        pytest.importorskip("jax")
        config["general"]["enable logging"] = False
        config["general"]["jax"] = True
        config["general"]["random state seed"] = 42

        track = Track()

        energy = 100.0
        ratio = track.additional_track_ratio(energy, "total")
        assert np.isfinite(ratio)
        assert ratio >= 0
