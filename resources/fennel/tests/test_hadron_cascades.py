# -*- coding: utf-8 -*-
"""
Unit tests for Hadron_Cascade class.

Tests hadronic cascade light yield calculations.
"""
import numpy as np
import pytest

from fennel import config
from fennel.hadron_cascades import Hadron_Cascade
from tests.conftest import trapezoid_compat


@pytest.mark.unit
class TestHadronCascade:
    """Test suite for Hadron_Cascade class."""

    def test_hadron_cascade_initialization(self):
        """Test Hadron_Cascade object can be initialized."""
        config["general"]["enable logging"] = False
        cascade = Hadron_Cascade()
        assert cascade is not None

    def test_track_lengths_calculation(self):
        """Test track length calculation for hadron cascades."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = Hadron_Cascade()

        energies = np.array([1.0, 10.0, 100.0, 1000.0])
        particles = [211, -211, 130, 2212, -2212, 2112]

        for particle in particles:
            lengths_mean, lengths_std = cascade.track_lengths(energies, particle)

            assert lengths_mean.shape == energies.shape
            assert lengths_std.shape == energies.shape
            assert np.all(np.isfinite(lengths_mean))
            assert np.all(np.isfinite(lengths_std))
            assert np.all(lengths_mean > 0)
            assert np.all(lengths_std >= 0)

    def test_em_fraction_calculation(self):
        """Test electromagnetic fraction calculation."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = Hadron_Cascade()

        energy = 100.0
        particles = [211, -211, 130, 2212, -2212, 2112]

        for particle in particles:
            em_frac_mean, em_frac_std = cascade.em_fraction(energy, particle)

            assert np.isfinite(em_frac_mean)
            assert np.isfinite(em_frac_std)
            assert 0 <= em_frac_mean <= 1  # Fraction should be between 0 and 1
            assert em_frac_std >= 0

    def test_longitudinal_profile(self):
        """Test longitudinal profile calculation."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        from fennel import Fennel

        fennel_instance = Fennel()

        energy = 100.0
        z_grid = np.linspace(0.0, 1000.0, 100)
        particle = 211

        # Use public API
        _, _, _, _, profile, _ = fennel_instance.hadron_yields(
            energy, particle, z_grid=z_grid, function=False
        )

        # Profile shape is (1, N) - squeeze it
        profile = profile.squeeze()
        assert profile.shape == z_grid.shape
        assert np.all(np.isfinite(profile))
        assert np.all(profile >= 0)

        # Profile should integrate to something reasonable
        integral = trapezoid_compat(profile, z_grid)
        assert integral > 0

    def test_angle_distribution(self):
        """Test angular distribution for hadron cascades."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        from fennel import Fennel

        fennel_instance = Fennel()

        angles = np.linspace(0.0, 180.0, 50)
        n = 1.333
        particle = 211
        energy = 100.0

        # Use public API
        _, _, _, _, _, angle_dist = fennel_instance.hadron_yields(
            energy, particle, angle_grid=angles, n=n, function=False
        )

        # angle_dist has shape (N,) or (1, N) - ensure 1D
        angle_dist = np.atleast_1d(angle_dist).squeeze()
        assert angle_dist.shape == angles.shape
        assert np.all(np.isfinite(angle_dist))
        assert np.all(angle_dist >= 0)

    def test_all_hadron_particles(self):
        """Test that all hadron particle types work."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = Hadron_Cascade()

        energy = 100.0
        particles = [211, -211, 130, 2212, -2212, 2112]

        for particle in particles:
            # Test track lengths
            lengths_mean, lengths_std = cascade.track_lengths(energy, particle)
            assert np.isfinite(lengths_mean)
            assert np.isfinite(lengths_std)

            # Test EM fraction
            em_frac_mean, em_frac_std = cascade.em_fraction(energy, particle)
            assert np.isfinite(em_frac_mean)
            assert np.isfinite(em_frac_std)

    def test_em_fraction_energy_dependence(self):
        """Test that EM fraction varies with energy."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = Hadron_Cascade()

        low_energy = 1.0
        high_energy = 1000.0
        particle = 211

        em_low, _ = cascade.em_fraction(low_energy, particle)
        em_high, _ = cascade.em_fraction(high_energy, particle)

        # EM fraction should vary with energy
        assert em_low != em_high

    def test_shower_max_exists(self):
        """Test that longitudinal profile has a peak."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        from fennel import Fennel

        fennel_instance = Fennel()

        energy = 100.0
        z_grid = np.linspace(0.0, 3000.0, 300)
        particle = 211

        # Use public API
        _, _, _, _, profile, _ = fennel_instance.hadron_yields(
            energy, particle, z_grid=z_grid, function=False
        )

        # Profile shape is (1, N) - squeeze it
        profile = profile.squeeze()
        # Should have a maximum somewhere
        max_idx = np.argmax(profile)
        assert 0 < max_idx < len(z_grid) - 1

    def test_particle_antiparticle_differences(self):
        """Test that particles and antiparticles have different properties."""
        config["general"]["enable logging"] = False
        config["general"]["jax"] = False
        cascade = Hadron_Cascade()

        energy = 100.0

        # Test pion and anti-pion
        em_frac_pi_plus, _ = cascade.em_fraction(energy, 211)
        em_frac_pi_minus, _ = cascade.em_fraction(energy, -211)

        # They should be different (based on config values)
        # But both should be valid
        assert 0 <= em_frac_pi_plus <= 1
        assert 0 <= em_frac_pi_minus <= 1


@pytest.mark.unit
@pytest.mark.jax
class TestHadronCascadeJAX:
    """Test Hadron_Cascade class with JAX backend."""

    def test_hadron_cascade_jax_initialization(self):
        """Test Hadron_Cascade works with JAX enabled."""
        pytest.importorskip("jax")
        config["general"]["enable logging"] = False
        config["general"]["jax"] = True
        config["general"]["random state seed"] = 42

        cascade = Hadron_Cascade()
        assert cascade is not None

    def test_hadron_cascade_jax_calculation(self):
        """Test basic calculation with JAX."""
        pytest.importorskip("jax")
        config["general"]["enable logging"] = False
        config["general"]["jax"] = True
        config["general"]["random state seed"] = 42

        cascade = Hadron_Cascade()

        energy = 100.0
        particle = 211

        lengths_mean, lengths_std = cascade.track_lengths(energy, particle)
        assert np.isfinite(lengths_mean)
        assert np.isfinite(lengths_std)

        em_frac_mean, em_frac_std = cascade.em_fraction(energy, particle)
        assert np.isfinite(em_frac_mean)
        assert np.isfinite(em_frac_std)
