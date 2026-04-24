# distrax and dm-haiku removed.
#
# Replacements:
#   haiku MLP   → pure-JAX _mlp_apply() reading weights directly from the
#                 existing pickle param dict (key scheme unchanged: no migration).
#   distrax RQS → _rqs_fwd / _rqs_inv vendored from distrax (Apache-2.0).
#                 Identical parameterisation → identical numerical outputs.
#   distrax Gamma log_prob → analytic formula in pure JAX.
#   distrax Gamma sample   → jax.random.gamma.
#   Training functions retain optax (already a direct dependency).
#
# Pickle files are read with the same key structure as before; no migration needed.
#
# RQS vendoring notice:
#   The functions _normalize_bin_sizes, _normalize_knot_slopes, _rqs_fwd,
#   _rqs_inv, and _safe_quadratic_root are adapted from
#   distrax/_src/bijectors/rational_quadratic_spline.py
#   (Copyright 2021 DeepMind Technologies Limited, Apache License 2.0).

import jax
import jax.numpy as jnp
import optax

# ---------------------------------------------------------------------------
# Pure-JAX MLP
# Reads directly from the haiku-style param dict:
#   params["mlp/~/linear_0"]["w"]  shape (in_dim, hidden)
#   params["mlp/~/linear_0"]["b"]  shape (hidden,)
#   ...
#   params["linear"]["w"]          shape (hidden, out)
#   params["linear"]["b"]          shape (out,)
# Hidden activation: ReLU (same as haiku MLP default).
# ---------------------------------------------------------------------------


def _mlp_apply(params, x, n_hidden_layers):
    """Apply a simple MLP using parameters in haiku-style dicts.

    Parameters
    ----------
    params : dict
        Parameter dictionary following the haiku naming scheme (weights and biases).
    x : jax.numpy.ndarray
        Input array of shape (..., in_dim).
    n_hidden_layers : int
        Number of hidden layers to apply.

    Returns
    -------
    jax.numpy.ndarray
        Network output array.
    """
    for i in range(n_hidden_layers):
        w = params[f"mlp/~/linear_{i}"]["w"]
        b = params[f"mlp/~/linear_{i}"]["b"]
        x = jax.nn.relu(x @ w + b)
    w = params["linear"]["w"]
    b = params["linear"]["b"]
    return x @ w + b


# ---------------------------------------------------------------------------
# Rational-quadratic spline helpers
# Adapted from distrax/_src/bijectors/rational_quadratic_spline.py
# (Copyright 2021 DeepMind Technologies Limited, Apache License 2.0)
# ---------------------------------------------------------------------------


def _normalize_bin_sizes(unnormalized, total_size, min_bin_size=1e-4):
    """
    Softmax-normalise bin sizes.

    Each bin >= min_bin_size, sum = total_size.

    Parameters
    ----------
    unnormalized : jax.numpy.ndarray
        Unnormalized bin sizes.
    total_size : float
        Total size to normalize to.
    min_bin_size : float, optional
        Minimum bin size (default is 1e-4).

    Returns
    -------
    jax.numpy.ndarray
        Normalized bin sizes.
    """
    num_bins = unnormalized.shape[-1]
    bin_sizes = jax.nn.softmax(unnormalized, axis=-1)
    return bin_sizes * (total_size - num_bins * min_bin_size) + min_bin_size


def _normalize_knot_slopes(unnormalized, min_knot_slope=1e-4):
    """
    Softplus-normalise knot slopes.

    Each slope >= min_knot_slope.

    Parameters
    ----------
    unnormalized : jax.numpy.ndarray
        Unnormalized knot slopes.
    min_knot_slope : float, optional
        Minimum knot slope (default is 1e-4).

    Returns
    -------
    jax.numpy.ndarray
        Normalized knot slopes.
    """
    # Offset chosen so that unnormalized=0 → normalized slope = 1.
    offset = jnp.log(jnp.exp(1.0 - min_knot_slope) - 1.0)
    return jax.nn.softplus(unnormalized + offset) + min_knot_slope


def _build_rqs_knots_1d(spl_p, rmin, rmax):
    """
    Build knot arrays from a 1-D (unbatched) spline parameter vector.

    Parameters
    ----------
    spl_p : jax.numpy.ndarray
        Spline parameters, shape (3*num_bins + 1,).
    rmin : float
        Minimum range value.
    rmax : float
        Maximum range value.

    Returns
    -------
    x_pos : jax.numpy.ndarray
        Knot positions in x, shape (num_bins + 1,).
    y_pos : jax.numpy.ndarray
        Knot positions in y, shape (num_bins + 1,).
    knot_slopes : jax.numpy.ndarray
        Knot slopes, shape (num_bins + 1,).
    """
    num_bins = (spl_p.shape[0] - 1) // 3
    range_size = rmax - rmin
    bin_widths = _normalize_bin_sizes(spl_p[:num_bins], range_size)
    bin_heights = _normalize_bin_sizes(spl_p[num_bins : 2 * num_bins], range_size)
    knot_slopes = _normalize_knot_slopes(spl_p[2 * num_bins :])
    # Interior knot positions (distrax convention: explicit rmin / rmax endpoints).
    x_pos = jnp.concatenate(
        [
            jnp.array([rmin]),
            rmin + jnp.cumsum(bin_widths[:-1]),
            jnp.array([rmax]),
        ]
    )
    y_pos = jnp.concatenate(
        [
            jnp.array([rmin]),
            rmin + jnp.cumsum(bin_heights[:-1]),
            jnp.array([rmax]),
        ]
    )
    return x_pos, y_pos, knot_slopes


def _build_rqs_knots_batched(spl_params, rmin, rmax):
    """
    Build knot arrays from a batched spline parameter matrix.

    Parameters
    ----------
    spl_params : jax.numpy.ndarray
        Array with shape (batch, 3*num_bins + 1).
    rmin : float
        Minimum range value.
    rmax : float
        Maximum range value.

    Returns
    -------
    tuple
        Tuple of arrays ``(x_pos, y_pos, knot_slopes)`` each with shape
        (batch, num_bins + 1).
    """
    num_bins = (spl_params.shape[-1] - 1) // 3
    range_size = rmax - rmin
    bin_widths = _normalize_bin_sizes(spl_params[..., :num_bins], range_size)
    bin_heights = _normalize_bin_sizes(spl_params[..., num_bins : 2 * num_bins], range_size)
    knot_slopes = _normalize_knot_slopes(spl_params[..., 2 * num_bins :])
    pad_shape = spl_params.shape[:-1] + (1,)
    pad_min = jnp.full(pad_shape, rmin)
    pad_max = jnp.full(pad_shape, rmax)
    x_pos = jnp.concatenate(
        [
            pad_min,
            rmin + jnp.cumsum(bin_widths[..., :-1], axis=-1),
            pad_max,
        ],
        axis=-1,
    )
    y_pos = jnp.concatenate(
        [
            pad_min,
            rmin + jnp.cumsum(bin_heights[..., :-1], axis=-1),
            pad_max,
        ],
        axis=-1,
    )
    return x_pos, y_pos, knot_slopes


def _rqs_fwd(x, x_pos, y_pos, knot_slopes):
    """
    Rational-quadratic spline forward pass for a single scalar x.

    Parameters
    ----------
    x : float
        Input scalar.
    x_pos, y_pos, knot_slopes : jax.numpy.ndarray
        1-D arrays of shape (num_bins + 1,).

    Returns
    -------
    tuple
        ``(y, log_det)`` where ``y`` is the transformed scalar and ``log_det`` is
        the log absolute derivative ``log|dy/dx|``.
    """
    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]
    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate([jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]).astype(bool)
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_l = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_r = jnp.sum(correct_bin[:, None] * params[1:], axis=0)
    x0, x1 = params_l[0], params_r[0]
    y0, y1 = params_l[1], params_r[1]
    d0, d1 = params_l[2], params_r[2]

    bin_w = x1 - x0
    bin_h = y1 - y0
    s = bin_h / bin_w
    z = jnp.clip((x - x0) / bin_w, 0.0, 1.0)
    z2 = z * z
    z1mz = z - z2  # z(1-z)
    z1m2 = (1.0 - z) ** 2

    st = d1 + d0 - 2.0 * s
    num = bin_h * (s * z2 + d0 * z1mz)
    den = s + st * z1mz
    y = y0 + num / den

    log_det = 2.0 * jnp.log(s) + jnp.log(d1 * z2 + 2.0 * s * z1mz + d0 * z1m2) - 2.0 * jnp.log(den)

    # Linear extrapolation outside [rmin, rmax]
    y = jnp.where(below_range, (x - x_pos[0]) * knot_slopes[0] + y_pos[0], y)
    y = jnp.where(above_range, (x - x_pos[-1]) * knot_slopes[-1] + y_pos[-1], y)
    log_det = jnp.where(below_range, jnp.log(knot_slopes[0]), log_det)
    log_det = jnp.where(above_range, jnp.log(knot_slopes[-1]), log_det)
    return y, log_det


def _safe_quadratic_root(a, b, c):
    """Numerically stable root of az² + bz + c = 0 for z ∈ [0,1].

    Parameters
    ----------
    a : float or jax.numpy.ndarray
        Quadratic coefficient.
    b : float or jax.numpy.ndarray
        Linear coefficient.
    c : float or jax.numpy.ndarray
        Constant term.

    Returns
    -------
    jax.numpy.ndarray
        Numerically stable root value (intended for z in [0, 1]).
    """
    sqrt_diff = b**2 - 4.0 * a * c
    safe_sqrt = jnp.sqrt(jnp.clip(sqrt_diff, jnp.finfo(sqrt_diff.dtype).tiny))
    safe_sqrt = jnp.where(sqrt_diff > 0.0, safe_sqrt, 0.0)
    num = jnp.where(b >= 0, 2.0 * c, -b + safe_sqrt)
    den = jnp.where(b >= 0, -b - safe_sqrt, 2.0 * a)
    return num / den


def _rqs_inv(y, x_pos, y_pos, knot_slopes):
    """
    Rational-quadratic spline inverse pass for a single scalar y.

    Parameters
    ----------
    y : float
        Input scalar in the target space.
    x_pos, y_pos, knot_slopes : jax.numpy.ndarray
        1-D arrays of shape (num_bins + 1,).

    Returns
    -------
    tuple
        ``(x, log_det)`` where ``x`` is the inverse-transformed scalar and
        ``log_det`` is the log absolute derivative ``log|dx/dy|``.
    """
    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]
    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate([jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]).astype(bool)
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_l = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_r = jnp.sum(correct_bin[:, None] * params[1:], axis=0)
    x0, x1 = params_l[0], params_r[0]
    y0, y1 = params_l[1], params_r[1]
    d0, d1 = params_l[2], params_r[2]

    bin_w = x1 - x0
    bin_h = y1 - y0
    s = bin_h / bin_w
    w = jnp.clip((y - y0) / bin_h, 0.0, 1.0)
    st = d1 + d0 - 2.0 * s
    c_ = -s * w
    b_ = d0 - st * w
    a_ = s - b_
    z = jnp.clip(_safe_quadratic_root(a_, b_, c_), 0.0, 1.0)
    x = bin_w * z + x0

    z2 = z * z
    z1mz = z - z2
    z1m2 = (1.0 - z) ** 2
    den = s + st * z1mz
    log_det = -2.0 * jnp.log(s) - jnp.log(d1 * z2 + 2.0 * s * z1mz + d0 * z1m2) + 2.0 * jnp.log(den)

    x = jnp.where(below_range, (y - y_pos[0]) / knot_slopes[0] + x_pos[0], x)
    x = jnp.where(above_range, (y - y_pos[-1]) / knot_slopes[-1] + x_pos[-1], x)
    log_det = jnp.where(below_range, -jnp.log(knot_slopes[0]), log_det)
    log_det = jnp.where(above_range, -jnp.log(knot_slopes[-1]), log_det)
    return x, log_det


# ---------------------------------------------------------------------------
# Gamma distribution helpers (pure JAX)
# ---------------------------------------------------------------------------


def _gamma_log_prob(x, concentration=1.5, rate=0.1):
    """Log probability of x under Gamma(concentration, rate).

    Parameters
    ----------
    x : float or jax.numpy.ndarray
        Value(s) at which to evaluate the log-probability.
    concentration : float, optional
        Shape (concentration) parameter (default is 1.5).
    rate : float, optional
        Rate parameter (default is 0.1).

    Returns
    -------
    jax.numpy.ndarray
        Log-probability evaluated at ``x``.
    """
    return (
        (concentration - 1.0) * jnp.log(x)
        - rate * x
        - jax.lax.lgamma(concentration)
        + concentration * jnp.log(rate)
    )


def _gamma_sample(key, concentration=1.5, rate=0.1, shape=()):
    """Sample from Gamma(concentration, rate).

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key for sampling.
    concentration : float, optional
        Shape parameter of the Gamma distribution.
    rate : float, optional
        Rate parameter of the Gamma distribution.
    shape : tuple, optional
        Output sample shape.

    Returns
    -------
    jax.numpy.ndarray
        Samples from the Gamma distribution scaled by ``1 / rate``.
    """
    return jax.random.gamma(key, a=concentration, shape=shape) / rate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class _ConditionerFn:
    """Drop-in replacement for hk.Transformed with .apply(params, x)."""

    def __init__(self, n_hidden_layers):
        """Initialize conditioner wrapper.

        Parameters
        ----------
        n_hidden_layers : int
            Number of hidden layers in the underlying MLP.
        """
        self._n = n_hidden_layers

    def apply(self, params, x):
        """Apply the underlying MLP to inputs using given parameters.

        Parameters
        ----------
        params : dict
            Haiku-style parameter dictionary.
        x : jax.numpy.ndarray
            Input array with shape (..., in_dim).

        Returns
        -------
        jax.numpy.ndarray
            Network output array.
        """
        return _mlp_apply(params, x, self._n)


def make_conditioner(hidden_sizes, out_params_activ, init_zero=True):
    """Conditioner MLP factory (kept for training API compatibility).

    Parameters
    ----------
    hidden_sizes : sequence
        Hidden layer sizes for the MLP.
    out_params_activ : callable or None
        Activation applied to output parameters (kept for API compatibility).
    init_zero : bool, optional
        If True, initialise output parameters to zero.

    Returns
    -------
    _ConditionerFn
        Callable conditioner object with an ``.apply(params, x)`` method.
    """
    return _ConditionerFn(n_hidden_layers=len(hidden_sizes))


def make_shape_conditioner_fn(mlp_hidden_size, mlp_num_layers, flow_num_bins, flow_num_layers):
    """Build the shape-model conditioner (MLP).

    Parameters
    ----------
    mlp_hidden_size : int
        Hidden layer size for the MLP.
    mlp_num_layers : int
        Number of hidden layers.
    flow_num_bins : int
        Number of bins used in the flow (kept for API compatibility).
    flow_num_layers : int
        Number of flow layers (kept for API compatibility).

    Returns
    -------
    _ConditionerFn
        Conditioner callable.
    """
    return _ConditionerFn(n_hidden_layers=mlp_num_layers)


def make_spl_flow(spl_params_list, rmin, rmax):
    """Convert a list of raw spline parameter arrays into knot-tuple lists.

    Parameters
    ----------
    spl_params_list : sequence of jax.numpy.ndarray
        List of spline parameter arrays; each element has shape (batch, 3*num_bins + 1).
    rmin : float
        Minimum range value for the spline.
    rmax : float
        Maximum range value for the spline.

    Returns
    -------
    list
        List of tuples ``(x_pos, y_pos, knot_slopes)`` each with shape
        (batch, num_bins + 1).
    """
    return [_build_rqs_knots_batched(sp, float(rmin), float(rmax)) for sp in spl_params_list]


class _TrafDistBuilder:
    """Callable matching the original traf_dist_builder(…) API.

    Calling an instance returns either a TransformedDist (return_base=False)
    or a (BaseDist, Flow) pair (return_base=True).
    """

    def __init__(self, flow_num_layers, rmin, rmax, return_base):
        """Initialise the traffic distribution builder.

        Parameters
        ----------
        flow_num_layers : int
            Number of spline layers in the flow.
        rmin : float
            Minimum range value for the spline.
        rmax : float
            Maximum range value for the spline.
        return_base : bool
            If True, ``__call__`` returns ``(base_dist, flow)``.
        """
        self.flow_num_layers = flow_num_layers
        self.rmin = float(rmin)
        self.rmax = float(rmax)
        self.return_base = return_base

    def __call__(self, traf_params):
        """Build and return the distribution object for given parameters.

        Parameters
        ----------
        traf_params : jax.numpy.ndarray
            Array of transformed flow parameters.

        Returns
        -------
        object
            Depending on ``return_base`` either a dist-like object with
            ``.log_prob()`` or a ``(base_dist, flow)`` pair.
        """
        return self._make(traf_params)

    def _make(self, traf_params):
        """Construct flow-related helper classes for given traffic parameters.

        Parameters
        ----------
        traf_params : jax.numpy.ndarray
            Flattened flow parameter vector.

        Returns
        -------
        object
            Depending on ``self.return_base`` either a ``_TransformedDist``
            instance or a ``(_BaseDist, _Flow)`` pair.
        """
        spl_params_list = jnp.split(traf_params, self.flow_num_layers, axis=-1)
        spline_layers = make_spl_flow(spl_params_list, self.rmin, self.rmax)
        builder = self

        class _Flow:
            """Implements Inverse(Chain([spl0, spl1, …, shift4])).forward(z).

            distrax Chain.forward applies LAST bijector first.
            Chain([spl0,spl1,shift4]).inverse(z) applies FIRST bijector first:
              x1 = spl0.inv(z)
              x2 = spl1.inv(x1)
              y  = shift4.inv(x2) = x2 - 4
            Inverse(Chain).forward = Chain.inverse, so:
              y = spl1.inv(spl0.inv(z)) - 4
            """

            def forward(self, z):
                """Forward transformation of the flow on input ``z``.

                Parameters
                ----------
                z : array-like
                    Input samples in the flow target space.

                Returns
                -------
                array-like
                    Inverse-transformed samples.
                """
                fn_inv = jnp.vectorize(_rqs_inv, signature="(),(n),(n),(n)->(),()")
                x = z
                for xp, yp, ks in spline_layers:  # spl0 first, then spl1, …
                    x, _ = fn_inv(x, xp, yp, ks)
                x = x - 4.0  # shift4.inverse
                return x

        class _BaseDist:
            """Simple base distribution wrapper exposing ``sample()``."""

            def sample(self, seed, sample_shape=()):
                """Draw samples from the base Gamma distribution.

                Parameters
                ----------
                seed : jax.random.PRNGKey
                    PRNG key for sampling.
                sample_shape : tuple, optional
                    Output sample shape.

                Returns
                -------
                jax.numpy.ndarray
                    Samples from the base distribution.
                """
                return _gamma_sample(seed, concentration=1.5, rate=0.1, shape=sample_shape)

        class _TransformedDist:
            """Distribution-like object exposing ``log_prob()``."""

            def log_prob(self, samples):
                """Compute log-probability of ``samples`` under the flow.

                Parameters
                ----------
                samples : array-like
                    Samples in the target space.

                Returns
                -------
                jax.numpy.ndarray
                    Log-probabilities for each sample.
                """
                return builder.log_prob(traf_params, samples)

        if self.return_base:
            return _BaseDist(), _Flow()
        return _TransformedDist()

    def log_prob(self, traf_params, samples):
        """Compute log p(y) for each (traf_params[i], samples[i]) pair.

        bijector = Inverse(Chain([spl0, spl1, shift4]))
        distrax Chain.forward applies LAST bijector first, so:
          Chain.forward(y) = spl0(spl1(shift4(y))) = spl0(spl1(y+4))
          log_det = ldj_shift4 + ldj_spl1(y+4) + ldj_spl0(spl1(y+4))
                  = 0 + ldj_spl1(y+4) + ldj_spl0(spl1(y+4))
        log p(y) = log p_Gamma(spl0(spl1(y+4))) + ldj_spl1 + ldj_spl0
        """
        rmin, rmax = self.rmin, self.rmax
        flow_num_layers = self.flow_num_layers

        def _single(tp, s):
            """Compute log-probability for a single parameter/sample pair.

            Parameters
            ----------
            tp : array-like
                Flattened spline parameter vector for one instance.
            s : float
                Scalar sample value.

            Returns
            -------
            jax.numpy.ndarray
                Log-probability scalar.
            """
            # tp: (total_params,)   s: scalar
            spl_p_list = jnp.split(tp, flow_num_layers)
            y = s + 4.0  # shift4.forward
            total_ldj = jnp.zeros(())
            # Chain.forward applies last-to-first: spl1 before spl0
            for spl_p in reversed(spl_p_list):
                xp, yp, ks = _build_rqs_knots_1d(spl_p, rmin, rmax)
                y_new, ld = _rqs_fwd(y, xp, yp, ks)
                total_ldj = total_ldj + ld
                y = y_new
            z = y
            return _gamma_log_prob(z) + total_ldj

        return jax.vmap(_single)(traf_params, samples)


def traf_dist_builder(flow_num_layers, flow_range, return_base=False):
    """
    Return a callable that builds the transformed distribution.

    Parameters
    ----------
    flow_num_layers : int
        Number of spline layers in the flow.
    flow_range : tuple
        ``(rmin, rmax)`` range for the spline.
    return_base : bool, optional
        If True, calling the returned object returns ``(base_dist, flow)``.
        If False, returns a dist-like object with ``.log_prob()``.

    Returns
    -------
    _TrafDistBuilder
        Builder callable for the transformed distribution.
    """
    return _TrafDistBuilder(
        flow_num_layers=flow_num_layers,
        rmin=flow_range[0],
        rmax=flow_range[1],
        return_base=return_base,
    )


def eval_log_prob(dist_builder, traf_params, samples):
    """
    Compute log p(samples | traf_params) under the flow.

    Parameters
    ----------
    dist_builder : _TrafDistBuilder or callable
        Object returned by :func:`traf_dist_builder`.
    traf_params : jax.numpy.ndarray
        Array with shape (batch, total_flow_params).
    samples : jax.numpy.ndarray
        Array with shape (batch,) of sample values.

    Returns
    -------
    jax.numpy.ndarray
        Log-probabilities with shape (batch,).
    """
    if isinstance(dist_builder, _TrafDistBuilder):
        return dist_builder.log_prob(traf_params, samples)
    # Fallback: call builder and use its .log_prob()
    return dist_builder(traf_params).log_prob(samples)


def sample_shape_model(dist_builder, traf_params, n_photons, seed):
    """
    Sample from the shape model.

    Parameters
    ----------
    dist_builder : callable
        Builder returned by :func:`traf_dist_builder` with ``return_base=True``.
    traf_params : jax.numpy.ndarray
        Array with shape (batch, total_flow_params).
    n_photons : int or tuple
        Number of base samples to draw or sample shape.
    seed : jax.random.PRNGKey
        JAX PRNG key.

    Returns
    -------
    jax.numpy.ndarray
        Samples drawn from the transformed shape model.
    """
    base_dist, trafo = dist_builder(traf_params)
    base_samples = base_dist.sample(seed=seed, sample_shape=n_photons)
    return trafo.forward(base_samples)


def make_counts_net_fn(config):
    """Build the counts-model MLP."""
    return _ConditionerFn(n_hidden_layers=config["mlp_num_layers"])


# ---------------------------------------------------------------------------
# Training functions — use optax; hk.PRNGSequence replaced with jax.random
# ---------------------------------------------------------------------------


def _prng_seq(seed):
    """Infinite generator of fresh JAX PRNG keys.

    Parameters
    ----------
    seed : int
        Integer seed used to initialise the JAX PRNG.

    Yields
    ------
    jax.random.PRNGKey
        Infinite sequence of JAX PRNG subkeys.
    """
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def _init_mlp_params(in_dim, hidden_size, n_hidden, out_dim, seed):
    """He-initialised param dict matching the haiku key scheme.

    Parameters
    ----------
    in_dim : int
        Input dimensionality.
    hidden_size : int
        Width of hidden layers.
    n_hidden : int
        Number of hidden layers.
    out_dim : int
        Output dimensionality.
    seed : int
        Random seed for parameter initialisation.

    Returns
    -------
    dict
        Parameter dictionary following the haiku naming/key convention.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    params = {}
    d = in_dim
    for i in range(n_hidden):
        scale = np.sqrt(2.0 / d)
        params[f"mlp/~/linear_{i}"] = {
            "w": jnp.array(rng.normal(scale=scale, size=(d, hidden_size)), dtype=jnp.float32),
            "b": jnp.zeros(hidden_size, dtype=jnp.float32),
        }
        d = hidden_size
    params["linear"] = {
        "w": jnp.zeros((d, out_dim), dtype=jnp.float32),
        "b": jnp.zeros(out_dim, dtype=jnp.float32),
    }
    return params


def train_shape_model(config, train_loader, test_loader, seed=1337, writer=None):
    """Train the shape model using the provided data loaders.

    Parameters
    ----------
    config : dict
        Training and model configuration dictionary.
    train_loader : iterable
        Training data loader.
    test_loader : iterable
        Test data loader.
    seed : int, optional
        Random seed for parameter initialisation (default is 1337).
    writer : SummaryWriter or None, optional
        Optional writer for logging metrics.

    Returns
    -------
    dict
        Trained parameter dictionary.
    """

    dist_builder = traf_dist_builder(
        config["flow_num_layers"],
        (config["flow_rmin"], config["flow_rmax"]),
    )
    shape_conditioner = make_shape_conditioner_fn(
        config["mlp_hidden_size"],
        config["mlp_num_layers"],
        config["flow_num_bins"],
        config["flow_num_layers"],
    )

    @jax.jit
    def ema_update(params, avg_params):
        """Exponential moving average update for parameters.

        Parameters
        ----------
        params : dict
            Current parameters.
        avg_params : dict
            Current EMA parameters.

        Returns
        -------
        dict
            Updated EMA parameters.
        """
        return optax.incremental_update(params, avg_params, step_size=0.001)

    @jax.jit
    def loss_fn(params, cond, samples):
        """Compute negative log-likelihood loss for a batch.

        Parameters
        ----------
        params : dict
            Model parameters for the conditioner MLP.
        cond : array-like
            Conditioning inputs.
        samples : array-like
            Observed sample values.

        Returns
        -------
        jax.numpy.ndarray
            Scalar loss value.
        """
        traf_params = shape_conditioner.apply(params, cond)
        lprobs = eval_log_prob(dist_builder, traf_params, samples)
        return -jnp.mean(lprobs * jnp.isfinite(lprobs))

    @jax.jit
    def update(params, opt_state, cond, samples):
        """Perform a single optimization step.

        Parameters
        ----------
        params : dict
            Current model parameters.
        opt_state : optax.OptState
            Current optimizer state.
        cond : array-like
            Conditioning inputs for the batch.
        samples : array-like
            Observed samples for the batch.

        Returns
        -------
        tuple
            ``(new_params, new_opt_state, loss_value)``.
        """
        lval, grads = jax.value_and_grad(loss_fn)(params, cond, samples)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state, lval

    scheduler = optax.cosine_decay_schedule(config["lr"], config["steps"], alpha=0.0)
    optimizer = optax.adam(learning_rate=scheduler)

    n_out = (3 * config["flow_num_bins"] + 1) * config["flow_num_layers"]
    params = avg_params = _init_mlp_params(
        2, config["mlp_hidden_size"], config["mlp_num_layers"], n_out, seed
    )
    opt_state = optimizer.init(params)

    train_iter = iter(train_loader)
    for i in range(1, config["steps"] + 1):
        train = next(train_iter)
        cond = jnp.concatenate(train[:2]).T
        samples = jnp.squeeze(train[2])
        params, opt_state, train_loss = update(params, opt_state, cond, samples)
        avg_params = ema_update(params, avg_params)

        if i % 100 == 0:
            test_loss = (
                sum(
                    loss_fn(avg_params, jnp.concatenate(t[:2]).T, jnp.squeeze(t[2]))
                    for t in test_loader
                )
                / test_loader._n_batches
            )
            train_loss, test_loss = jax.device_get((train_loss, test_loss))
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, i)
                writer.add_scalar("Loss/test", test_loss, i)
                writer.flush()
            print(f"Epoch: {i} \t Train/Test: {train_loss:.3E} / {test_loss:.3E}")

    return params


def train_counts_model(config, train_loader, test_loader, seed=1337, writer=None):
    """Train the counts model using MLP regression.

    Parameters
    ----------
    config : dict
        Training and model configuration.
    train_loader : iterable
        Training data loader.
    test_loader : iterable
        Test data loader.
    seed : int, optional
        Random seed.
    writer : SummaryWriter or None, optional
        Optional writer for logging metrics.

    Returns
    -------
    dict
        Trained parameter dictionary.
    """

    net_fn = make_counts_net_fn(config)

    @jax.jit
    def loss_fn(params, batch):
        """Mean-squared error loss for a batch.

        Parameters
        ----------
        params : dict
            Model parameters.
        batch : tuple
            Batch data (inputs, targets, ...).

        Returns
        -------
        jax.numpy.ndarray
            Scalar loss value.
        """
        inp = jnp.concatenate(batch[:2]).T
        out = net_fn.apply(params, inp).squeeze()
        return 0.5 * jnp.average((out - batch[2]) ** 2)

    @jax.jit
    def update(params, opt_state, batch):
        """Single optimizer update for counts model.

        Returns updated parameters, optimizer state and loss.
        """
        lval, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state, lval

    scheduler = optax.cosine_decay_schedule(config["lr"], config["steps"], alpha=0.0)
    optimizer = optax.adam(learning_rate=scheduler)

    params = _init_mlp_params(2, config["mlp_hidden_size"], config["mlp_num_layers"], 1, seed)
    opt_state = optimizer.init(params)

    train_iter = iter(train_loader)
    for i in range(1, config["steps"] + 1):
        train = next(train_iter)
        params, opt_state, train_loss = update(params, opt_state, train)

        if i % 100 == 0:
            test_loss = sum(loss_fn(params, t) for t in test_loader) / test_loader._n_batches
            train_loss, test_loss = jax.device_get((train_loss, test_loss))
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, i)
                writer.add_scalar("Loss/test", test_loss, i)
                writer.flush()
            print(f"Epoch: {i} \t Train/Test: {train_loss:.3E} / {test_loss:.3E}")

    if writer is not None:
        test_loss = sum(loss_fn(params, t) for t in test_loader) / test_loader._n_batches
        test_loss = jax.device_get(test_loss)
        writer.add_hparams(dict(config), {"hparam/test_loss": test_loss})
        writer.flush()
        writer.close()

    return params
