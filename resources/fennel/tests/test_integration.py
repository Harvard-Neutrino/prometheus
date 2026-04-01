# -*- coding: utf-8 -*-
"""
Integration tests for the Fennel class.

Tests the full workflow and public API.
"""
import numpy as np
import pytest

from tests.conftest import trapezoid_compat


@pytest.mark.integration
class TestFennelIntegration:
    """Integration tests for Fennel class."""

    def test_fennel_initialization(self, fennel_instance):
        """Test Fennel can be initialized and closed properly."""
        assert fennel_instance is not None

    def test_track_yields_muon(self, fennel_instance, test_wavelengths):
        """Test track yields for muon."""
        energy = 100.0

        dcounts, angles = fennel_instance.track_yields(
            energy, wavelengths=test_wavelengths, function=False
        )

        assert dcounts.shape == test_wavelengths.shape
        assert np.all(np.isfinite(dcounts))
        assert np.all(dcounts >= 0)

    def test_track_yields_function_mode(self, fennel_instance):
        """Test track yields in function mode."""
        energy = 100.0

        dcounts_func, angles_func = fennel_instance.track_yields(energy, function=True)

        # Should return callable functions
        assert callable(dcounts_func)
        assert callable(angles_func)

        # Test calling the functions
        wavelengths = np.linspace(350.0, 500.0, 10)
        angles = np.linspace(0.0, 180.0, 10)
        n = 1.333

        dcounts_val = dcounts_func(energy, wavelengths)
        angles_val = angles_func(angles, n, energy)

        assert dcounts_val.shape == wavelengths.shape
        assert angles_val.shape == angles.shape

    def test_em_yields_electron(self, fennel_instance, test_wavelengths, test_z_grid):
        """Test EM yields for electron."""
        energy = 100.0
        particle = 11  # electron

        dcounts, dcounts_sample, long_profile, angles = fennel_instance.em_yields(
            energy,
            particle,
            wavelengths=test_wavelengths,
            z_grid=test_z_grid,
            function=False,
        )

        assert dcounts.shape == test_wavelengths.shape
        assert dcounts_sample.shape == test_wavelengths.shape
        # long_profile has shape (1, N) - squeeze to compare
        long_profile = long_profile.squeeze()
        assert long_profile.shape == test_z_grid.shape
        assert np.all(np.isfinite(dcounts))
        assert np.all(np.isfinite(dcounts_sample))
        assert np.all(np.isfinite(long_profile))

    def test_em_yields_all_particles(self, fennel_instance, em_particles):
        """Test EM yields for all EM particles."""
        energy = 100.0

        for particle in em_particles:
            dcounts, dcounts_sample, long_profile, angles = fennel_instance.em_yields(
                energy, particle, function=False
            )

            assert np.all(np.isfinite(dcounts))
            assert np.all(np.isfinite(dcounts_sample))
            assert np.all(np.isfinite(long_profile))

    def test_hadron_yields_pion(self, fennel_instance, test_wavelengths, test_z_grid):
        """Test hadron yields for pion."""
        energy = 100.0
        particle = 211  # pi+

        dcounts, dcounts_sample, em_frac, em_frac_sample, long_profile, angles = (
            fennel_instance.hadron_yields(
                energy,
                particle,
                wavelengths=test_wavelengths,
                z_grid=test_z_grid,
                function=False,
            )
        )

        assert dcounts.shape == test_wavelengths.shape
        assert dcounts_sample.shape == test_wavelengths.shape
        # long_profile has shape (1, N) - squeeze to compare
        long_profile = long_profile.squeeze()
        assert long_profile.shape == test_z_grid.shape
        assert np.all(np.isfinite(dcounts))
        assert np.all(np.isfinite(dcounts_sample))
        assert np.all(np.isfinite(long_profile))
        assert np.isfinite(em_frac)
        assert np.isfinite(em_frac_sample)

    def test_hadron_yields_all_particles(self, fennel_instance, hadron_particles):
        """Test hadron yields for all hadron particles."""
        energy = 100.0

        for particle in hadron_particles:
            dcounts, dcounts_sample, em_frac, em_frac_sample, long_profile, angles = (
                fennel_instance.hadron_yields(energy, particle, function=False)
            )

            assert np.all(np.isfinite(dcounts))
            assert np.all(np.isfinite(dcounts_sample))
            assert np.all(np.isfinite(long_profile))
            assert np.isfinite(em_frac)
            assert np.isfinite(em_frac_sample)

    def test_auto_yields_track(self, fennel_instance):
        """Test auto_yields correctly routes to track."""
        energy = 100.0
        particle = 13  # muon

        dcounts, dcounts_s, em_frac, em_frac_s, long, angles = (
            fennel_instance.auto_yields(energy, particle, function=False)
        )

        assert dcounts is not None
        assert angles is not None
        # These should be None for tracks
        assert dcounts_s is None
        assert em_frac is None
        assert em_frac_s is None
        assert long is None

    def test_auto_yields_em(self, fennel_instance):
        """Test auto_yields correctly routes to EM cascade."""
        energy = 100.0
        particle = 11  # electron

        dcounts, dcounts_s, em_frac, em_frac_s, long, angles = (
            fennel_instance.auto_yields(energy, particle, function=False)
        )

        assert dcounts is not None
        assert dcounts_s is not None
        assert long is not None
        assert angles is not None
        # EM fraction should be None for EM cascades
        assert em_frac is None
        assert em_frac_s is None

    def test_auto_yields_hadron(self, fennel_instance):
        """Test auto_yields correctly routes to hadron cascade."""
        energy = 100.0
        particle = 211  # pion

        dcounts, dcounts_s, em_frac, em_frac_s, long, angles = (
            fennel_instance.auto_yields(energy, particle, function=False)
        )

        # All should be populated for hadrons
        assert dcounts is not None
        assert dcounts_s is not None
        assert em_frac is not None
        assert em_frac_s is not None
        assert long is not None
        assert angles is not None

    def test_multiple_energies(self, fennel_instance, test_energies):
        """Test handling of multiple energy values."""
        for energy in test_energies:
            dcounts, angles = fennel_instance.track_yields(energy, function=False)
            assert np.all(np.isfinite(dcounts))

    def test_interaction_types(self, fennel_instance):
        """Test different interaction types for tracks."""
        energy = 100.0
        interactions = ["ionization", "pair", "brems", "nuclear", "total"]

        for interaction in interactions:
            dcounts, angles = fennel_instance.track_yields(
                energy, interaction=interaction, function=False
            )
            assert np.all(np.isfinite(dcounts))

    def test_invalid_particle(self, fennel_instance):
        """Test that invalid particle raises appropriate error."""
        energy = 100.0
        invalid_particle = 99999

        with pytest.raises(ValueError):
            fennel_instance.auto_yields(energy, invalid_particle)

    def test_photon_counts_positive(self, fennel_instance, test_wavelengths):
        """Test that photon counts are always positive."""
        energy = 100.0

        # Test track
        dcounts_track, _ = fennel_instance.track_yields(
            energy, wavelengths=test_wavelengths, function=False
        )
        assert np.all(dcounts_track >= 0)

        # Test EM
        dcounts_em, _, _, _ = fennel_instance.em_yields(
            energy, 11, wavelengths=test_wavelengths, function=False
        )
        assert np.all(dcounts_em >= 0)

        # Test hadron
        dcounts_had, _, _, _, _, _ = fennel_instance.hadron_yields(
            energy, 211, wavelengths=test_wavelengths, function=False
        )
        assert np.all(dcounts_had >= 0)

    def test_integral_counts(self, fennel_instance, test_wavelengths):
        """Test that integrated counts are reasonable."""
        energy = 100.0

        dcounts, _ = fennel_instance.track_yields(
            energy, wavelengths=test_wavelengths, function=False
        )

        total_counts = trapezoid_compat(dcounts, test_wavelengths)
        assert total_counts > 0
        assert np.isfinite(total_counts)
