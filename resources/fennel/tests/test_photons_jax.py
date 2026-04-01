import pytest

from fennel import Fennel, config


@pytest.mark.jax
def test_photon_track_jax_build_and_call():
    """Ensure Photon track functions are jitted without argname mismatch."""
    pytest.importorskip("jax")
    config["general"]["jax"] = True
    f = Fennel()
    try:
        # Request functional forms (JAX mode returns callables accepting scalars)
        dcounts_func, angles_func = f.track_yields(1.0, function=True)
        # Call the returned functions with scalar inputs to ensure jit works
        wavelengths = config["advanced"]["wavelengths"]
        n = config["mediums"][config["scenario"]["medium"]]["refractive index"]
        # Call dcounts_func for a single wavelength
        dcounts_func(1.0, wavelengths[0])
        # Call angles function: signature (angle, n, energy)
        angles_func(0.0, n, 1.0)
    finally:
        f.close()
