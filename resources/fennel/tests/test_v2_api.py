# -*- coding: utf-8 -*-
"""
Tests for v2.0 API enhancements.

These tests verify the new result containers, validation, and convenience
methods while ensuring backward compatibility.
"""

import numpy as np
import pytest

from fennel import (
    EMYieldResult,
    Fennel,
    HadronYieldResult,
    TrackYieldResult,
    ValidationError,
)


class TestResultContainers:
    """Test result container classes."""

    def setup_method(self):
        """Create Fennel instance for tests."""
        self.fennel = Fennel()

    def teardown_method(self):
        """Cleanup after tests."""
        self.fennel.close()

    def test_track_result_container(self):
        """Test TrackYieldResult container."""
        result = self.fennel.track_yields_v2(100.0)

        assert isinstance(result, TrackYieldResult)
        assert result.energy == 100.0
        assert result.interaction == "total"
        assert isinstance(result.dcounts, np.ndarray)
        assert isinstance(result.angles, np.ndarray)
        assert not result.is_function
        assert "TrackYieldResult" in repr(result)

    def test_em_result_container(self):
        """Test EMYieldResult container."""
        result = self.fennel.em_yields_v2(1000.0, particle=11)

        assert isinstance(result, EMYieldResult)
        assert result.energy == 1000.0
        assert result.particle == 11
        assert result.particle_name == "electron"
        assert isinstance(result.dcounts, np.ndarray)
        assert isinstance(result.longitudinal_profile, np.ndarray)
        assert not result.is_function
        assert "electron" in repr(result)

    def test_hadron_result_container(self):
        """Test HadronYieldResult container."""
        result = self.fennel.hadron_yields_v2(1000.0, particle=211)

        assert isinstance(result, HadronYieldResult)
        assert result.energy == 1000.0
        assert result.particle == 211
        assert result.particle_name == "π+"
        assert isinstance(result.em_fraction, float)
        assert 0.0 < result.em_fraction < 1.0
        assert isinstance(result.dcounts, np.ndarray)
        assert "π+" in repr(result)

    def test_function_mode(self):
        """Test result containers in function mode."""
        result = self.fennel.track_yields_v2(100.0, function=True)

        assert result.is_function
        assert callable(result.dcounts)
        assert callable(result.angles)
        assert "functional" in repr(result)


class TestConvenienceMethods:
    """Test convenience methods."""

    def setup_method(self):
        """Create Fennel instance for tests."""
        self.fennel = Fennel()

    def teardown_method(self):
        """Cleanup after tests."""
        self.fennel.close()

    def test_quick_track(self):
        """Test quick_track convenience method."""
        result = self.fennel.quick_track(100.0)

        assert isinstance(result, TrackYieldResult)
        assert result.energy == 100.0
        assert result.interaction == "total"

    def test_quick_track_with_interaction(self):
        """Test quick_track with custom interaction."""
        result = self.fennel.quick_track(100.0, interaction="brems")

        assert result.interaction == "brems"

    def test_quick_cascade_em(self):
        """Test quick_cascade with EM particle."""
        result = self.fennel.quick_cascade(100.0, particle=11)

        assert isinstance(result, EMYieldResult)
        assert result.particle == 11

    def test_quick_cascade_hadron(self):
        """Test quick_cascade with hadron particle."""
        result = self.fennel.quick_cascade(100.0, particle=211)

        assert isinstance(result, HadronYieldResult)
        assert result.particle == 211

    def test_calculate_with_pdg(self):
        """Test calculate method with PDG ID."""
        # Test track
        result = self.fennel.calculate(100.0, particle=13)
        assert isinstance(result, TrackYieldResult)

        # Test EM cascade
        result = self.fennel.calculate(100.0, particle=11)
        assert isinstance(result, EMYieldResult)

        # Test hadron cascade
        result = self.fennel.calculate(100.0, particle=211)
        assert isinstance(result, HadronYieldResult)

    def test_calculate_with_type_name(self):
        """Test calculate method with particle type name."""
        result = self.fennel.calculate(100.0, particle_type="muon")
        assert isinstance(result, TrackYieldResult)

        result = self.fennel.calculate(100.0, particle_type="electron")
        assert isinstance(result, EMYieldResult)

        result = self.fennel.calculate(100.0, particle_type="pion")
        assert isinstance(result, HadronYieldResult)


class TestValidation:
    """Test input validation."""

    def setup_method(self):
        """Create Fennel instance for tests."""
        self.fennel = Fennel()

    def teardown_method(self):
        """Cleanup after tests."""
        self.fennel.close()

    def test_negative_energy_error(self):
        """Test validation rejects negative energy."""
        with pytest.raises(ValidationError) as exc_info:
            self.fennel.track_yields_v2(-100.0)
        assert "positive" in str(exc_info.value).lower()

    def test_zero_energy_error(self):
        """Test validation rejects zero energy."""
        with pytest.raises(ValidationError) as exc_info:
            self.fennel.track_yields_v2(0.0)
        assert "positive" in str(exc_info.value).lower()

    def test_invalid_particle_error(self):
        """Test validation rejects unknown particle."""
        with pytest.raises(ValidationError) as exc_info:
            self.fennel.em_yields_v2(100.0, particle=999)
        assert "Unknown particle" in str(exc_info.value)

    def test_invalid_interaction_error(self):
        """Test validation rejects unknown interaction."""
        with pytest.raises(ValidationError) as exc_info:
            self.fennel.track_yields_v2(100.0, interaction="invalid")
        assert "Unknown interaction" in str(exc_info.value)

    def test_invalid_wavelengths_error(self):
        """Test validation rejects invalid wavelengths."""
        with pytest.raises(ValidationError) as exc_info:
            self.fennel.track_yields_v2(100.0, wavelengths=np.array([]))
        assert "empty" in str(exc_info.value).lower()

    def test_helpful_error_messages(self):
        """Test that error messages include helpful suggestions."""
        with pytest.raises(ValidationError) as exc_info:
            self.fennel.track_yields_v2(-100.0)
        error_msg = str(exc_info.value)
        assert "Example:" in error_msg  # Should include example usage


class TestBackwardCompatibility:
    """Test that old API still works (backward compatibility)."""

    def setup_method(self):
        """Create Fennel instance for tests."""
        self.fennel = Fennel()

    def teardown_method(self):
        """Cleanup after tests."""
        self.fennel.close()

    def test_old_track_yields(self):
        """Test old track_yields API still works."""
        dcounts, angles = self.fennel.track_yields(100.0)

        assert isinstance(dcounts, np.ndarray)
        assert isinstance(angles, np.ndarray)
        assert len(dcounts.shape) == 1
        assert len(angles.shape) == 1

    def test_old_em_yields(self):
        """Test old em_yields API still works."""
        dcounts, dcounts_s, long_prof, angles = self.fennel.em_yields(
            1000.0, particle=11
        )

        assert isinstance(dcounts, np.ndarray)
        assert isinstance(dcounts_s, np.ndarray)
        assert isinstance(long_prof, np.ndarray)
        assert isinstance(angles, np.ndarray)

    def test_old_hadron_yields(self):
        """Test old hadron_yields API still works."""
        dcounts, dcounts_s, em_frac, em_frac_s, long_prof, angles = (
            self.fennel.hadron_yields(1000.0, particle=211)
        )

        assert isinstance(dcounts, np.ndarray)
        assert isinstance(em_frac, float)
        assert 0.0 < em_frac < 1.0

    def test_v2_and_old_api_equivalence(self):
        """Test that v2 and old API produce same results."""
        # Track yields
        old_dcounts, old_angles = self.fennel.track_yields(100.0)
        new_result = self.fennel.track_yields_v2(100.0)

        np.testing.assert_array_equal(old_dcounts, new_result.dcounts)
        np.testing.assert_array_equal(old_angles, new_result.angles)

        # EM yields - note: dcounts_sample uses randomness so check shape only
        old_dc, old_dcs, old_lp, old_ang = self.fennel.em_yields(100.0, 11)
        new_result = self.fennel.em_yields_v2(100.0, 11)

        np.testing.assert_array_equal(old_dc, new_result.dcounts)
        # dcounts_sample is stochastic, just check shape
        assert old_dcs.shape == new_result.dcounts_sample.shape
        np.testing.assert_array_equal(old_lp, new_result.longitudinal_profile)
        np.testing.assert_array_equal(old_ang, new_result.angles)
