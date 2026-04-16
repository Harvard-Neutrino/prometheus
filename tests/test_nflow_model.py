"""Regression tests for the normalizing-flow and counts neural-net models.

These tests lock in the numerical outputs of the current haiku/distrax
implementation so that the Phase 7 migration to a replacement backend can be
validated against them.  A passing test suite here is the *definition* of a
correct migration.

All reference values were captured from the trained P-ONE pickle files with
JAX PRNGKey(42) on CPU (April 2026).  The tolerance is generous (1e-4) because
float32 ordering between implementations can introduce small differences; the
physics does not care about sub-percent timing shifts.

Test inputs are three (log10_distance, angle_rad) pairs covering the usable
range of the model:
    row 0:  log10(10 m),  0.5 rad  —  short distance
    row 1:  log10(50 m),  1.2 rad  —  mid distance
    row 2:  log10(200 m), 2.1 rad  —  long distance (near the log10(300) mask)
"""
import pickle
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prometheus.compat.haiku_unpickler import load as haiku_load

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RESOURCES = pathlib.Path("resources/olympus_resources")
SHAPE_PICKLE = RESOURCES / "photon_arrival_time_nflow_params.pickle"
COUNTS_PICKLE = RESOURCES / "photon_arrival_time_counts_params.pickle"

# Fixed test inputs: shape (3, 2) — [log10_dist, angle]
TEST_INPUTS = jnp.array([
    [1.0, 0.5],
    [1.7, 1.2],
    [2.3, 2.1],
], dtype=jnp.float32)

# Tolerance for float32 comparisons across backends
ATOL = 1e-4


@pytest.fixture(scope="module")
def shape_model():
    config, params = haiku_load(SHAPE_PICKLE)
    return config, params


@pytest.fixture(scope="module")
def counts_model():
    config, params = haiku_load(COUNTS_PICKLE)
    return config, params


# ---------------------------------------------------------------------------
# Pickle structure tests — these must pass before AND after migration
# ---------------------------------------------------------------------------

def test_shape_pickle_loads():
    config, params = haiku_load(SHAPE_PICKLE)
    assert isinstance(config, dict)
    assert isinstance(params, dict)


def test_counts_pickle_loads():
    config, params = haiku_load(COUNTS_PICKLE)
    assert isinstance(config, dict)
    assert isinstance(params, dict)


def test_shape_config_keys(shape_model):
    config, _ = shape_model
    required = {"flow_num_layers", "flow_num_bins", "flow_rmin", "flow_rmax",
                "mlp_hidden_size", "mlp_num_layers"}
    assert required <= set(config.keys())


def test_counts_config_keys(counts_model):
    config, _ = counts_model
    required = {"mlp_hidden_size", "mlp_num_layers"}
    assert required <= set(config.keys())


def test_shape_params_keys(shape_model):
    """Top-level param keys follow the haiku mlp/~/linear_N naming scheme."""
    _, params = shape_model
    assert "linear" in params
    assert "mlp/~/linear_0" in params


def test_counts_params_keys(counts_model):
    _, params = counts_model
    assert "linear" in params
    assert "mlp/~/linear_0" in params


def test_shape_params_weight_shapes(shape_model):
    config, params = shape_model
    h = config["mlp_hidden_size"]
    # First hidden layer: input is 2 features
    assert params["mlp/~/linear_0"]["w"].shape == (2, h)
    assert params["mlp/~/linear_0"]["b"].shape == (h,)


def test_counts_params_weight_shapes(counts_model):
    config, params = counts_model
    h = config["mlp_hidden_size"]
    assert params["mlp/~/linear_0"]["w"].shape == (2, h)
    assert params["mlp/~/linear_0"]["b"].shape == (h,)


# ---------------------------------------------------------------------------
# Shape conditioner (MLP) regression
# ---------------------------------------------------------------------------

def test_shape_conditioner_output_shape(shape_model):
    from hyperion.models.photon_arrival_time_nflow.net import make_shape_conditioner_fn
    config, params = shape_model
    conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"], config["mlp_num_layers"],
        config["flow_num_bins"], config["flow_num_layers"],
    )
    out = conditioner.apply(params, TEST_INPUTS)
    num_bijector_params = 3 * config["flow_num_bins"] + 1
    expected_cols = num_bijector_params * config["flow_num_layers"]
    assert out.shape == (3, expected_cols)


def test_shape_conditioner_regression(shape_model):
    """Lock in first 6 columns of traf_params for each test row."""
    from hyperion.models.photon_arrival_time_nflow.net import make_shape_conditioner_fn
    config, params = shape_model
    conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"], config["mlp_num_layers"],
        config["flow_num_bins"], config["flow_num_layers"],
    )
    out = np.array(conditioner.apply(params, TEST_INPUTS))

    ref = np.array([
        [-1.440378,  0.48451898, 1.6659592, 1.486107,  1.4062678, 1.3814201],
        [-1.497236, -0.01775454, 2.1847098, 1.834826,  1.6428913, 1.603916 ],
        [-1.9864869,-0.69843817, 3.2488124, 2.7716594, 2.3765428, 2.2806304],
    ], dtype=np.float32)

    np.testing.assert_allclose(out[:, :6], ref, atol=ATOL)


# ---------------------------------------------------------------------------
# Flow log_prob regression (deterministic — no random key needed)
# ---------------------------------------------------------------------------

# Fixed photon arrival times within the flow's valid range [flow_rmin, flow_rmax]
_FIXED_SAMPLES = jnp.array([5.0, 50.0, 200.0], dtype=jnp.float32)


def test_flow_log_prob_regression(shape_model):
    """Lock in log_prob for fixed (traf_params, sample) pairs.

    This is the primary migration-validation test: it is fully deterministic
    (no random key) and exercises the entire bijector chain.
    """
    from hyperion.models.photon_arrival_time_nflow.net import (
        make_shape_conditioner_fn, traf_dist_builder, eval_log_prob,
    )
    config, params = shape_model
    conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"], config["mlp_num_layers"],
        config["flow_num_bins"], config["flow_num_layers"],
    )
    traf_params = conditioner.apply(params, TEST_INPUTS)
    dist_builder = traf_dist_builder(
        config["flow_num_layers"],
        (config["flow_rmin"], config["flow_rmax"]),
    )
    lp = np.array(eval_log_prob(dist_builder, traf_params, _FIXED_SAMPLES))

    ref = np.array([-4.329562, -5.9898934, -6.5038505], dtype=np.float32)
    np.testing.assert_allclose(lp, ref, atol=ATOL)


def test_flow_log_prob_shape(shape_model):
    from hyperion.models.photon_arrival_time_nflow.net import (
        make_shape_conditioner_fn, traf_dist_builder, eval_log_prob,
    )
    config, params = shape_model
    conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"], config["mlp_num_layers"],
        config["flow_num_bins"], config["flow_num_layers"],
    )
    traf_params = conditioner.apply(params, TEST_INPUTS)
    dist_builder = traf_dist_builder(
        config["flow_num_layers"],
        (config["flow_rmin"], config["flow_rmax"]),
    )
    lp = eval_log_prob(dist_builder, traf_params, _FIXED_SAMPLES)
    assert lp.shape == (3,)


def test_flow_sampler_runs(shape_model):
    """Smoke test: sampler runs without error and returns the right shape."""
    from hyperion.models.photon_arrival_time_nflow.net import (
        make_shape_conditioner_fn, traf_dist_builder, sample_shape_model,
    )
    config, params = shape_model
    conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"], config["mlp_num_layers"],
        config["flow_num_bins"], config["flow_num_layers"],
    )
    traf_params = conditioner.apply(params, TEST_INPUTS)
    dist_builder = traf_dist_builder(
        config["flow_num_layers"],
        (config["flow_rmin"], config["flow_rmax"]),
        return_base=True,
    )
    key = jax.random.PRNGKey(0)
    samples = sample_shape_model(dist_builder, traf_params, 3, key)
    assert samples.shape == (3,)


# ---------------------------------------------------------------------------
# Counts net regression
# ---------------------------------------------------------------------------

def test_counts_net_output_shape(counts_model):
    from hyperion.models.photon_arrival_time_nflow.net import make_counts_net_fn
    config, params = counts_model
    net = make_counts_net_fn(config)
    out = net.apply(params, TEST_INPUTS)
    assert out.shape == (3, 1)


def test_counts_net_regression(counts_model):
    """Lock in log10-survival-fraction predictions."""
    from hyperion.models.photon_arrival_time_nflow.net import make_counts_net_fn
    config, params = counts_model
    net = make_counts_net_fn(config)
    out = np.array(net.apply(params, TEST_INPUTS)).squeeze()

    ref = np.array([-4.725943, -6.9162498, -10.032677], dtype=np.float32)
    np.testing.assert_allclose(out, ref, atol=ATOL)


def test_counts_net_monotone_with_distance(counts_model):
    """Survival fraction must decrease (more negative) as distance increases."""
    from hyperion.models.photon_arrival_time_nflow.net import make_counts_net_fn
    config, params = counts_model
    net = make_counts_net_fn(config)
    out = np.array(net.apply(params, TEST_INPUTS)).squeeze()
    # Each row has larger log10_dist than previous
    assert out[0] > out[1] > out[2], "Log-survival fraction must decrease with distance"
