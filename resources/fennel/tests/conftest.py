# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for Fennel tests.
"""
import numpy as np
import pytest

from fennel import Fennel, config


def trapezoid_compat(y, x=None, dx=None, axis=-1):
    """
    Backward-compatible trapezoid integration helper.

    Works with NumPy <2.0 (trapz) and NumPy 2.0+ (trapezoid).
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    else:
        return np.trapz(y, x=x, dx=dx, axis=axis)


@pytest.fixture(scope="session")
def reset_config():
    """Reset config to default state after tests."""
    yield
    # Reset to defaults if needed


@pytest.fixture
def fennel_instance():
    """Create a fresh Fennel instance with fixed random seed for reproducibility."""
    config["general"]["random state seed"] = 42
    config["general"]["enable logging"] = False
    instance = Fennel()
    yield instance
    instance.close()


@pytest.fixture
def test_energies():
    """Standard test energies in GeV."""
    return np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])


@pytest.fixture
def test_wavelengths():
    """Standard test wavelengths in nm."""
    return np.linspace(350.0, 500.0, 50)


@pytest.fixture
def test_angles():
    """Standard test angles in degrees."""
    return np.linspace(0.0, 180.0, 50)


@pytest.fixture
def test_z_grid():
    """Standard z-grid for longitudinal profiles in cm."""
    return np.linspace(0.0, 1e4, 1000)


@pytest.fixture
def track_particles():
    """PDG IDs for track particles (muons)."""
    return [13, -13]


@pytest.fixture
def em_particles():
    """PDG IDs for EM particles."""
    return [11, -11, 22]


@pytest.fixture
def hadron_particles():
    """PDG IDs for hadron particles."""
    return [211, -211, 130, 2212, -2212, 2112]


@pytest.fixture
def reference_medium():
    """Standard medium parameters."""
    return {
        "name": "water",
        "refractive_index": 1.333,
        "density": 1.0,
        "radiation_length": 36.08,
    }


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    """Parametrize tests to run with both NumPy and JAX backends."""
    if request.param == "jax":
        try:
            import jax

            config["general"]["jax"] = True
        except ImportError:
            pytest.skip("JAX not installed")
    else:
        config["general"]["jax"] = False
    return request.param
