# -*- coding: utf-8 -*-
"""
Physics regression tests with reference values.

These tests ensure that the physics results remain constant across code changes.
Reference values are computed once with the current implementation and stored.
Any deviation from these values indicates a change in physics behavior.

IMPORTANT: These reference values should NOT be changed unless there is a
deliberate update to the physics parametrization or bug fix.
"""
import json
import os

import numpy as np
import pytest

from fennel import Fennel, config
from tests.conftest import trapezoid_compat

# Reference values computed with v1.3.4
# These are the GOLD STANDARD values that must be preserved
REFERENCE_VALUES = {
    # Track particles - multiple energies and interactions
    "track_muon_1GeV_total": {
        "energy": 1.0,
        "particle": 13,
        "interaction": "total",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    "track_muon_10GeV_total": {
        "energy": 10.0,
        "particle": 13,
        "interaction": "total",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    "track_muon_100GeV_total": {
        "energy": 100.0,
        "particle": 13,
        "interaction": "total",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    "track_muon_1TeV_total": {
        "energy": 1000.0,
        "particle": 13,
        "interaction": "total",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    "track_muon_10TeV_total": {
        "energy": 10000.0,
        "particle": 13,
        "interaction": "total",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    # Different interactions
    "track_muon_100GeV_ionization": {
        "energy": 100.0,
        "particle": 13,
        "interaction": "ionization",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    "track_muon_100GeV_brems": {
        "energy": 100.0,
        "particle": 13,
        "interaction": "brems",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    "track_muon_100GeV_pair": {
        "energy": 100.0,
        "particle": 13,
        "interaction": "pair",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    # Anti-muon
    "track_antimuon_100GeV_total": {
        "energy": 100.0,
        "particle": -13,
        "interaction": "total",
        "wavelength": 400.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    # Different wavelengths
    "track_muon_100GeV_350nm": {
        "energy": 100.0,
        "particle": 13,
        "interaction": "total",
        "wavelength": 350.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    "track_muon_100GeV_450nm": {
        "energy": 100.0,
        "particle": 13,
        "interaction": "total",
        "wavelength": 450.0,
        "expected_dcounts": None,
        "tolerance": 1e-10,
    },
    # EM cascades - multiple energies
    "em_electron_1GeV": {
        "energy": 1.0,
        "particle": 11,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_electron_10GeV": {
        "energy": 10.0,
        "particle": 11,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_electron_100GeV": {
        "energy": 100.0,
        "particle": 11,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_electron_1TeV": {
        "energy": 1000.0,
        "particle": 11,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_electron_10TeV": {
        "energy": 10000.0,
        "particle": 11,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_positron_100GeV": {
        "energy": 100.0,
        "particle": -11,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_positron_1TeV": {
        "energy": 1000.0,
        "particle": -11,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_gamma_10GeV": {
        "energy": 10.0,
        "particle": 22,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_gamma_100GeV": {
        "energy": 100.0,
        "particle": 22,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_gamma_1TeV": {
        "energy": 1000.0,
        "particle": 22,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    # Different wavelengths for EM
    "em_electron_100GeV_350nm": {
        "energy": 100.0,
        "particle": 11,
        "wavelength": 350.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "em_electron_100GeV_500nm": {
        "energy": 100.0,
        "particle": 11,
        "wavelength": 500.0,
        "expected_dcounts": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    # Hadron cascades - pions
    "hadron_pion_plus_10GeV": {
        "energy": 10.0,
        "particle": 211,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_pion_plus_100GeV": {
        "energy": 100.0,
        "particle": 211,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_pion_plus_1TeV": {
        "energy": 1000.0,
        "particle": 211,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_pion_minus_10GeV": {
        "energy": 10.0,
        "particle": -211,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_pion_minus_100GeV": {
        "energy": 100.0,
        "particle": -211,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_pion_minus_1TeV": {
        "energy": 1000.0,
        "particle": -211,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    # Kaon
    "hadron_kaon_100GeV": {
        "energy": 100.0,
        "particle": 130,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_kaon_1TeV": {
        "energy": 1000.0,
        "particle": 130,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    # Protons
    "hadron_proton_10GeV": {
        "energy": 10.0,
        "particle": 2212,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_proton_100GeV": {
        "energy": 100.0,
        "particle": 2212,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_proton_1TeV": {
        "energy": 1000.0,
        "particle": 2212,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_proton_10TeV": {
        "energy": 10000.0,
        "particle": 2212,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    # Anti-protons
    "hadron_antiproton_100GeV": {
        "energy": 100.0,
        "particle": -2212,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_antiproton_1TeV": {
        "energy": 1000.0,
        "particle": -2212,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    # Neutrons
    "hadron_neutron_100GeV": {
        "energy": 100.0,
        "particle": 2112,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
    "hadron_neutron_1TeV": {
        "energy": 1000.0,
        "particle": 2112,
        "wavelength": 400.0,
        "expected_dcounts": None,
        "expected_em_fraction": None,
        "expected_integral": None,
        "tolerance": 1e-10,
    },
}


def save_reference_values(filename="tests/reference_values_v1.3.4.json"):
    """
    Generate and save reference values from current implementation.

    This should be run ONCE with the validated v1.3.4 code to establish
    the gold standard values.
    """
    config["general"]["random state seed"] = 42
    config["general"]["enable logging"] = False
    fennel = Fennel()

    wavelengths = np.linspace(300.0, 600.0, 200)

    print(f"Generating reference values for {len(REFERENCE_VALUES)} test cases...")

    # Process all test cases
    for key, ref in REFERENCE_VALUES.items():
        wavelength_idx = np.argmin(np.abs(wavelengths - ref["wavelength"]))

        if key.startswith("track_"):
            # Track yields
            dcounts, _ = fennel.track_yields(
                ref["energy"],
                wavelengths=wavelengths,
                interaction=ref["interaction"],
                function=False,
            )
            ref["expected_dcounts"] = float(dcounts[wavelength_idx])
            print(f"  {key}: dcounts={ref['expected_dcounts']:.6e}")

        elif key.startswith("em_"):
            # EM yields
            dcounts, _, _, _ = fennel.em_yields(
                ref["energy"], ref["particle"], wavelengths=wavelengths, function=False
            )
            ref["expected_dcounts"] = float(dcounts[wavelength_idx])
            ref["expected_integral"] = float(trapezoid_compat(dcounts, wavelengths))
            print(
                f"  {key}: dcounts={ref['expected_dcounts']:.6e}, integral={ref['expected_integral']:.6e}"
            )

        elif key.startswith("hadron_"):
            # Hadron yields
            dcounts, _, em_frac, _, _, _ = fennel.hadron_yields(
                ref["energy"], ref["particle"], wavelengths=wavelengths, function=False
            )
            ref["expected_dcounts"] = float(dcounts[wavelength_idx])
            ref["expected_em_fraction"] = float(em_frac)
            ref["expected_integral"] = float(trapezoid_compat(dcounts, wavelengths))
            print(
                f"  {key}: dcounts={ref['expected_dcounts']:.6e}, em_frac={ref['expected_em_fraction']:.4f}, integral={ref['expected_integral']:.6e}"
            )

    fennel.close()

    # Save to file
    with open(filename, "w") as f:
        json.dump(REFERENCE_VALUES, f, indent=2)

    print(f"\nReference values saved to {filename}")
    return REFERENCE_VALUES


def load_reference_values(filename="tests/reference_values_v1.3.4.json"):
    """Load reference values from file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None


@pytest.fixture(scope="module")
def reference_data():
    """Load or generate reference values."""
    refs = load_reference_values()
    if refs is None:
        pytest.skip(
            "Reference values not yet generated. Run save_reference_values() first."
        )
    return refs


@pytest.mark.physics
class TestPhysicsRegression:
    """Physics regression tests to ensure results don't change."""

    @pytest.mark.parametrize(
        "test_case",
        [key for key in REFERENCE_VALUES.keys() if key.startswith("track_")],
    )
    def test_track_yields(self, test_case, reference_data):
        """Test that track yields match reference values."""
        ref = reference_data[test_case]
        if ref["expected_dcounts"] is None:
            pytest.skip("Reference value not set")

        config["general"]["random state seed"] = 42
        config["general"]["enable logging"] = False
        fennel = Fennel()

        wavelengths = np.linspace(300.0, 600.0, 200)
        wavelength_idx = np.argmin(np.abs(wavelengths - ref["wavelength"]))

        dcounts, _ = fennel.track_yields(
            ref["energy"],
            wavelengths=wavelengths,
            interaction=ref["interaction"],
            function=False,
        )

        actual = dcounts[wavelength_idx]
        expected = ref["expected_dcounts"]

        fennel.close()

        assert (
            np.abs(actual - expected) < ref["tolerance"]
        ), f"{test_case}: Track yield changed! Expected {expected}, got {actual}"

    @pytest.mark.parametrize(
        "test_case", [key for key in REFERENCE_VALUES.keys() if key.startswith("em_")]
    )
    def test_em_yields(self, test_case, reference_data):
        """Test that EM cascade yields match reference values."""
        ref = reference_data[test_case]
        if ref["expected_dcounts"] is None:
            pytest.skip("Reference value not set")

        config["general"]["random state seed"] = 42
        config["general"]["enable logging"] = False
        fennel = Fennel()

        wavelengths = np.linspace(300.0, 600.0, 200)
        wavelength_idx = np.argmin(np.abs(wavelengths - ref["wavelength"]))

        dcounts, _, _, _ = fennel.em_yields(
            ref["energy"], ref["particle"], wavelengths=wavelengths, function=False
        )

        actual_dcounts = dcounts[wavelength_idx]
        actual_integral = trapezoid_compat(dcounts, wavelengths)

        fennel.close()

        assert (
            np.abs(actual_dcounts - ref["expected_dcounts"]) < ref["tolerance"]
        ), f"{test_case}: EM dcounts changed! Expected {ref['expected_dcounts']}, got {actual_dcounts}"
        assert (
            np.abs(actual_integral - ref["expected_integral"]) < ref["tolerance"]
        ), f"{test_case}: EM integral changed! Expected {ref['expected_integral']}, got {actual_integral}"

    @pytest.mark.parametrize(
        "test_case",
        [key for key in REFERENCE_VALUES.keys() if key.startswith("hadron_")],
    )
    def test_hadron_yields(self, test_case, reference_data):
        """Test that hadron cascade yields match reference values."""
        ref = reference_data[test_case]
        if ref["expected_dcounts"] is None:
            pytest.skip("Reference value not set")

        config["general"]["random state seed"] = 42
        config["general"]["enable logging"] = False
        fennel = Fennel()

        wavelengths = np.linspace(300.0, 600.0, 200)
        wavelength_idx = np.argmin(np.abs(wavelengths - ref["wavelength"]))

        dcounts, _, em_frac, _, _, _ = fennel.hadron_yields(
            ref["energy"], ref["particle"], wavelengths=wavelengths, function=False
        )

        actual_dcounts = dcounts[wavelength_idx]
        actual_em_frac = em_frac
        actual_integral = trapezoid_compat(dcounts, wavelengths)

        fennel.close()

        assert (
            np.abs(actual_dcounts - ref["expected_dcounts"]) < ref["tolerance"]
        ), f"{test_case}: Hadron dcounts changed! Expected {ref['expected_dcounts']}, got {actual_dcounts}"
        assert (
            np.abs(actual_em_frac - ref["expected_em_fraction"]) < ref["tolerance"]
        ), f"{test_case}: EM fraction changed! Expected {ref['expected_em_fraction']}, got {actual_em_frac}"
        assert (
            np.abs(actual_integral - ref["expected_integral"]) < ref["tolerance"]
        ), f"{test_case}: Hadron integral changed! Expected {ref['expected_integral']}, got {actual_integral}"

    @pytest.mark.slow
    def test_energy_range_consistency(self):
        """Test that yields are consistent across wide energy range."""
        config["general"]["random state seed"] = 42
        config["general"]["enable logging"] = False
        fennel = Fennel()

        energies = np.logspace(0, 4, 10)  # 1 GeV to 10 TeV
        wavelengths = np.linspace(350.0, 500.0, 50)

        prev_integral = None
        for energy in energies:
            dcounts, _ = fennel.track_yields(
                energy, wavelengths=wavelengths, function=False
            )
            integral = trapezoid_compat(dcounts, wavelengths)

            # Check monotonicity (higher energy -> more light)
            if prev_integral is not None:
                assert (
                    integral >= prev_integral * 0.99
                )  # Allow small numerical variations
            prev_integral = integral

        fennel.close()
