# -*- coding: utf-8 -*-
"""
Unit tests for EM_Cascade class.

Tests electromagnetic cascade light yield calculations.
"""
import numpy as np
import pytest

from fennel import config
from fennel.em_cascades import EM_Cascade
from tests.conftest import trapezoid_compat


@pytest.mark.unit
class TestEMCascade:
    """Test suite for EM_Cascade class."""

    def test_em_cascade_initialization(self):
        """Test EM_Cascade object can be initialized."""
        config["general"]["enable logging"] = False
        cascade = EM_Cascade()
        assert cascade is not None

    def test_track_lengths_calculation(self):
        """Test track length calculation for EM cascades."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = EM_Cascade()

        energies = np.array([1.0, 10.0, 100.0, 1000.0])
        particles = [11, -11, 22]  # e-, e+, gamma

        for particle in particles:
            lengths_mean, lengths_std = cascade.track_lengths(energies, particle)

            assert lengths_mean.shape == energies.shape
            assert lengths_std.shape == energies.shape
            assert np.all(np.isfinite(lengths_mean))
            assert np.all(np.isfinite(lengths_std))
            assert np.all(lengths_mean > 0)
            assert np.all(lengths_std >= 0)

    def test_longitudinal_profile(self):
        """Test longitudinal profile calculation."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        from fennel import Fennel

        fennel_instance = Fennel()

        energy = 100.0
        z_grid = np.linspace(0.0, 1000.0, 100)
        particle = 11

        # Use public API
        _, _, profile, _ = fennel_instance.em_yields(
            energy, particle, z_grid=z_grid, function=False
        )

        # Profile shape is (1, N) - squeeze to match z_grid
        profile = profile.squeeze()
        assert profile.shape == z_grid.shape
        assert np.all(np.isfinite(profile))
        assert np.all(profile >= 0)

        # Profile should integrate to something reasonable
        integral = trapezoid_compat(profile, z_grid)
        assert integral > 0

    def test_longitudinal_profile_peak(self):
        """Test that longitudinal profile has a peak (shower max)."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        from fennel import Fennel

        fennel_instance = Fennel()

        energy = 100.0
        z_grid = np.linspace(0.0, 2000.0, 200)
        particle = 11

        # Use public API
        _, _, profile, _ = fennel_instance.em_yields(
            energy, particle, z_grid=z_grid, function=False
        )

        # Profile shape is (1, N) - squeeze it
        profile = profile.squeeze()
        # Should have a maximum somewhere in the middle
        max_idx = np.argmax(profile)
        assert 0 < max_idx < len(z_grid) - 1

    def test_angle_distribution(self):
        """Test angular distribution for EM cascades."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = EM_Cascade()

        angles = np.linspace(0.0, 180.0, 50)
        n = 1.333
        particle = 11

        angle_dist = cascade.cherenkov_angle_distro(angles, n, particle)

        assert angle_dist.shape == angles.shape
        assert np.all(np.isfinite(angle_dist))
        assert np.all(angle_dist >= 0)

    def test_particle_types(self):
        """Test that all EM particle types work."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = EM_Cascade()

        energy = 100.0
        particles = [11, -11, 22]

        for particle in particles:
            lengths_mean, lengths_std = cascade.track_lengths(energy, particle)
            assert np.isfinite(lengths_mean)
            assert np.isfinite(lengths_std)

    def test_energy_scaling(self):
        """Test that track lengths scale with energy."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = EM_Cascade()

        low_energy = 10.0
        high_energy = 1000.0
        particle = 11

        low_length, _ = cascade.track_lengths(low_energy, particle)
        high_length, _ = cascade.track_lengths(high_energy, particle)

        # Higher energy should give longer tracks
        assert high_length > low_length

    def test_shower_max_energy_dependence(self):
        """Test that shower maximum position depends on energy."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        from fennel import Fennel

        fennel_instance = Fennel()

        z_grid = np.linspace(0.0, 3000.0, 300)
        particle = 11

        low_energy = 10.0
        high_energy = 1000.0

        # Use public API
        _, _, profile_low, _ = fennel_instance.em_yields(
            low_energy, particle, z_grid=z_grid, function=False
        )
        _, _, profile_high, _ = fennel_instance.em_yields(
            high_energy, particle, z_grid=z_grid, function=False
        )

        # Squeeze profiles
        profile_low = profile_low.squeeze()
        profile_high = profile_high.squeeze()

        max_low = z_grid[np.argmax(profile_low)]
        max_high = z_grid[np.argmax(profile_high)]

        # Shower max should be deeper for higher energy
        assert max_high > max_low


@pytest.mark.unit
@pytest.mark.jax
class TestEMCascadeJAX:
    """Test EM_Cascade class with JAX backend."""

    def test_em_cascade_jax_initialization(self):
        """Test EM_Cascade works with JAX enabled."""
        pytest.importorskip("jax")
        config["general"]["enable logging"] = False
        config["general"]["jax"] = True
        config["general"]["random state seed"] = 42

        cascade = EM_Cascade()
        assert cascade is not None

    def test_em_cascade_jax_calculation(self):
        """Test basic calculation with JAX."""
        pytest.importorskip("jax")
        config["general"]["enable logging"] = False
        config["general"]["jax"] = True
        config["general"]["random state seed"] = 42

        cascade = EM_Cascade()

        energy = 100.0
        particle = 11

        lengths_mean, lengths_std = cascade.track_lengths(energy, particle)
        assert np.isfinite(lengths_mean)
        assert np.isfinite(lengths_std)
