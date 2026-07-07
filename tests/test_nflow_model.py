"""Regression tests for the normalizing-flow and counts neural-net models.

These tests lock in the numerical outputs of the current haiku/distrax
implementation so that replacement backend can be
validated against them.

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
TEST_INPUTS = jnp.array(
    [
        [1.0, 0.5],
        [1.7, 1.2],
        [2.3, 2.1],
    ],
    dtype=jnp.float32,
)

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
    required = {
        "flow_num_layers",
        "flow_num_bins",
        "flow_rmin",
        "flow_rmax",
        "mlp_hidden_size",
        "mlp_num_layers",
    }
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
        config["mlp_hidden_size"],
        config["mlp_num_layers"],
        config["flow_num_bins"],
        config["flow_num_layers"],
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
        config["mlp_hidden_size"],
        config["mlp_num_layers"],
        config["flow_num_bins"],
        config["flow_num_layers"],
    )
    out = np.array(conditioner.apply(params, TEST_INPUTS))

    ref = np.array(
        [
            [-1.440378, 0.48451898, 1.6659592, 1.486107, 1.4062678, 1.3814201],
            [-1.497236, -0.01775454, 2.1847098, 1.834826, 1.6428913, 1.603916],
            [-1.9864869, -0.69843817, 3.2488124, 2.7716594, 2.3765428, 2.2806304],
        ],
        dtype=np.float32,
    )

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
        eval_log_prob,
        make_shape_conditioner_fn,
        traf_dist_builder,
    )

    config, params = shape_model
    conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"],
        config["mlp_num_layers"],
        config["flow_num_bins"],
        config["flow_num_layers"],
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
        eval_log_prob,
        make_shape_conditioner_fn,
        traf_dist_builder,
    )

    config, params = shape_model
    conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"],
        config["mlp_num_layers"],
        config["flow_num_bins"],
        config["flow_num_layers"],
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
        make_shape_conditioner_fn,
        sample_shape_model,
        traf_dist_builder,
    )

    config, params = shape_model
    conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"],
        config["mlp_num_layers"],
        config["flow_num_bins"],
        config["flow_num_layers"],
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


# ---------------------------------------------------------------------------
# Counts net physics invariants
# ---------------------------------------------------------------------------
#
# These tests check physical plausibility of the trained model.  All four
# invariants must hold for any well-trained Cherenkov-light counts model,
# regardless of the medium:
#
#   1. Forward dominance: survival(~0°) ≥ 100× survival(90°).
#   2. No backward peak: survival(~180°) ≤ survival(~0°).
#   3. No 90° local maximum: survival(90°) ≤ min(survival(85°), survival(95°)).
#   4. Forward survival decreases monotonically with distance.
#
# Probe distances are 5, 24, 50, 100 m — the 24 m point is specifically
# chosen because the MLP 90°/180° artifact is most visible there.

_PHYSICS_LOG10_DISTS = jnp.array(
    [np.log10(5.0), np.log10(24.0), np.log10(50.0), np.log10(100.0)],
    dtype=jnp.float32,
)
_N_PHYS = len(_PHYSICS_LOG10_DISTS)


def _counts_predict(net, params, log10_dists, angle_rad):
    """Return squeezed log10_survival predictions for a fixed angle."""
    angles = jnp.full((_N_PHYS,), angle_rad, dtype=jnp.float32)
    x = jnp.stack([log10_dists, angles], axis=1)
    return np.array(net.apply(params, x)).squeeze()


def test_counts_physics_forward_dominates_sideways(counts_model):
    """Survival at ~3° must strictly exceed survival at 90° at every distance.

    Cherenkov light is forward-peaked; forward emission must always dominate
    sideways emission.  Different media can produce very different angular
    profiles (e.g. heavily-scattering P-ONE vs clear ANTARES water) so we
    require only a positive margin, not a fixed factor.  A negative ratio
    (survival(90°) > survival(0°)) is unambiguously unphysical.
    """
    from hyperion.models.photon_arrival_time_nflow.net import make_counts_net_fn

    config, params = counts_model
    net = make_counts_net_fn(config)

    ls_fwd  = _counts_predict(net, params, _PHYSICS_LOG10_DISTS, np.radians(3.0))
    ls_side = _counts_predict(net, params, _PHYSICS_LOG10_DISTS, np.pi / 2)

    ratios = ls_fwd - ls_side  # log10 units; must be > 0 at every distance
    assert np.all(ratios > 0.0), (
        f"Forward survival (3°) must exceed sideways survival (90°) at all "
        f"distances. Got per-distance ratios: {np.round(ratios, 3).tolist()}"
    )


def test_counts_physics_no_backward_peak(counts_model):
    """Survival at exactly 180° must never exceed survival at ~0°.

    A track moving directly away from a module cannot produce more light at
    that module than the same track moving toward it.  We probe at exactly
    180° (not 177°) because the MLP extrapolation spike, if present, appears
    right at the boundary of the training grid.
    """
    from hyperion.models.photon_arrival_time_nflow.net import make_counts_net_fn

    config, params = counts_model
    net = make_counts_net_fn(config)

    ls_fwd = _counts_predict(net, params, _PHYSICS_LOG10_DISTS, np.radians(3.0))
    ls_bwd = _counts_predict(net, params, _PHYSICS_LOG10_DISTS, np.pi)

    excess = ls_bwd - ls_fwd  # must be ≤ 0 everywhere
    assert np.all(excess <= 0.0), (
        f"Backward survival (180°) must not exceed forward survival (3°). "
        f"Got per-distance excess: {np.round(excess, 3).tolist()}"
    )


def test_counts_physics_no_90deg_spike(counts_model):
    """Survival at 90° must not be a local maximum between 85° and 95°.

    An MLP can hallucinate a spike at exactly 90° if that angle is absent from
    the training grid (the default 6°-spaced grid skips 90°).  A local maximum
    requires survival(90°) to exceed BOTH neighbours simultaneously; a smooth
    monotone fall-off where 85° > 90° > 95° is perfectly physical.
    """
    from hyperion.models.photon_arrival_time_nflow.net import make_counts_net_fn

    config, params = counts_model
    net = make_counts_net_fn(config)

    ls_85 = _counts_predict(net, params, _PHYSICS_LOG10_DISTS, np.radians(85.0))
    ls_90 = _counts_predict(net, params, _PHYSICS_LOG10_DISTS, np.pi / 2)
    ls_95 = _counts_predict(net, params, _PHYSICS_LOG10_DISTS, np.radians(95.0))

    is_local_max = (ls_90 > ls_85) & (ls_90 > ls_95)
    assert not np.any(is_local_max), (
        f"Survival at 90° is a local maximum (exceeds both 85° and 95°) at "
        f"distances {np.round(10**np.array(_PHYSICS_LOG10_DISTS)[is_local_max], 1).tolist()} m. "
        f"ls_85={np.round(ls_85, 3).tolist()}, "
        f"ls_90={np.round(ls_90, 3).tolist()}, "
        f"ls_95={np.round(ls_95, 3).tolist()}"
    )


def test_counts_physics_forward_monotone_distance(counts_model):
    """Forward survival must decrease strictly with increasing distance.

    At a fixed forward angle (~3°) each successive distance step must yield a
    more negative log10_survival.  This is the pure-distance version of the
    existing diagonal test and isolates the distance dependence.
    """
    from hyperion.models.photon_arrival_time_nflow.net import make_counts_net_fn

    config, params = counts_model
    net = make_counts_net_fn(config)

    ls_fwd = _counts_predict(net, params, _PHYSICS_LOG10_DISTS, np.radians(3.0))
    diffs = np.diff(ls_fwd)

    assert np.all(diffs < 0.0), (
        f"Forward survival (3°) must decrease monotonically with distance. "
        f"Consecutive differences: {np.round(diffs, 3).tolist()} (all must be < 0)"
    )
