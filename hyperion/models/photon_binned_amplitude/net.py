"""This module is not used by any live code path in Prometheus/Olympus and
has no shipped model weights in ``resources/``.

It is kept here as a reference implementation of the binned-amplitude photon-yield approach (an earlier alternative to the normalizing-flow model in photon_arrival_time_nflow/).

It still depends on dm-haiku and has not been migrated to pure JAX.
"""
import functools
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hyperion.data import DataLoader


class HistMLP(hk.Module):
    """Histogram MLP module used as a small haiku model."""

    def __init__(self, output_size, layers, dropout, final_activations, name=None):
        """Initialise the Histogram MLP.

        Parameters
        ----------
        output_size : int
            Number of output units (histogram bins or channels).
        layers : sequence of int
            Hidden layer sizes.
        dropout : float
            Dropout rate applied during training.
        final_activations : callable or None
            Optional activation applied to final layer outputs.
        name : str, optional
            Module name passed to haiku.
        """
        super().__init__(name=name)
        self.output_size = output_size
        self.layers = layers
        self.dropout = dropout
        self.final_activations = final_activations

    def __call__(self, x, is_training):
        """Forward pass through the histogram MLP.

        Parameters
        ----------
        x : jnp.ndarray
            Input array with shape (..., in_dim).
        is_training : bool
            Whether the module is executed in training mode (applies dropout).

        Returns
        -------
        jnp.ndarray
            Output array with last dimension equal to ``output_size``.
        """
        for n_per_layer in self.layers:
            x = hk.Linear(n_per_layer)(x)
            # x = hk.BatchNorm(
            #     create_scale=True, create_offset=True, decay_rate=0.9
            # )(x, is_training=is_training)
            x = jax.nn.relu(x)
            if is_training:
                key = hk.next_rng_key()
                x = hk.dropout(key, self.dropout, x)

        x = hk.Linear(self.output_size)(x)

        if self.final_activations is not None:
            x = self.final_activations(x)

        return x


def make_forward_fn(conf):
    """Create a Haiku forward function for the histogram MLP.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing model parameters such as
        ``n_neurons`` and ``n_out``.

    Returns
    -------
    callable
        Forward function with signature ``(batch, is_training)``.
    """

    layers = [conf["n_neurons"], conf["n_neurons"], conf["n_neurons"]]

    def forward_fn(batch, is_training):
        """Forward function used during training.

        Parameters
        ----------
        batch : sequence
            Training batch where ``batch[0]`` contains inputs.
        is_training : bool
            Whether to run in training mode (enables dropout).

        Returns
        -------
        jnp.ndarray
            Model outputs for the batch.
        """
        inp = jnp.asarray(batch[0], dtype=jnp.float32)
        return HistMLP(conf["n_out"], layers, conf["dropout"], None)(inp, is_training)

    return forward_fn


def make_eval_forward_fn(conf):
    """Create an evaluation forward function (non-training) for the MLP.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing model parameters.

    Returns
    -------
    callable
        Forward function with signature ``(inp)``.
    """

    layers = [conf["n_neurons"], conf["n_neurons"], conf["n_neurons"]]

    def forward_fn(inp):
        """Evaluation forward function (no dropout).

        Parameters
        ----------
        inp : array-like
            Input array for evaluation.

        Returns
        -------
        jnp.ndarray
            Model outputs.
        """
        return HistMLP(conf["n_out"], layers, conf["dropout"], None)(inp, False)

    return forward_fn


def make_logp1_trafo(scale):
    """Create log(1 + x*scale) forward and inverse transformers.

    Parameters
    ----------
    scale : float
        Scaling applied before the log transform.

    Returns
    -------
    tuple
        ``(trafo, rev_trafo)`` functions for forward and reverse transforms.
    """

    def trafo(data):
        """Forward transform: log(1 + x*scale)."""
        return np.log(data * scale + 1)

    def rev_trafo(data):
        """Inverse of the forward transform."""
        return jnp.exp(data - 1) / scale

    return trafo, rev_trafo


def make_net_eval_from_pickle(path):
    """Load network parameters from a pickle and build an evaluation function.

    Parameters
    ----------
    path : str
        Path to the pickle file containing ``(params, state, conf, binning, trafo_scale)``.

    Returns
    -------
    tuple
        ``(net_eval_fn, binning)`` where ``net_eval_fn`` is a JIT-compiled callable
        that maps inputs to model outputs and ``binning`` is the histogram binning.
    """

    (params, state, conf, binning, trafo_scale) = pickle.load(open(path, "rb"))
    forward_fn = make_eval_forward_fn(conf)
    net = hk.transform_with_state(forward_fn)

    _, rev_trafo = make_logp1_trafo(trafo_scale)

    @jax.jit
    def net_eval_fn(x):
        """JIT-compiled evaluation function loading parameters from pickle.

        Parameters
        ----------
        x : array-like
            Model input.

        Returns
        -------
        array-like
            Model output after reverse transform.
        """
        return rev_trafo(net.apply(params, state, None, x)[0])

    return net_eval_fn, binning


def train_net(conf, train_data, test_data, writer, rng):
    """Train the histogram MLP network.

    Parameters
    ----------
    conf : dict
        Model and training configuration.
    train_data : dataset-like
        Training dataset.
    test_data : dataset-like
        Test dataset.
    writer : SummaryWriter or None
        Optional writer for logging metrics.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple
        ``(net_eval_fn, avg_params, state)`` where ``net_eval_fn`` is a callable
        for model evaluation and ``avg_params``/``state`` are the trained params.
    """

    train_loader = DataLoader(
        train_data,
        batch_size=conf["batch_size"],
        shuffle=True,
        # worker_init_fn=seed_worker,
        rng=rng,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=conf["batch_size"],
        shuffle=False,
        # worker_init_fn=seed_worker,
        rng=rng,
    )

    forward_fn = make_forward_fn(conf)

    net = hk.transform_with_state(forward_fn)
    key = hk.PRNGSequence(42)

    params, state = net.init(next(key), next(iter(train_loader)), is_training=True)
    avg_params = params

    schedule = optax.cosine_decay_schedule(
        conf["lr"], conf["epochs"] * train_loader.n_batches, alpha=0.0
    )

    opt = optax.adam(learning_rate=schedule)
    opt_state = opt.init(params)

    def loss(params, state, rng_key, batch, is_training):
        """Compute loss for a batch during training.

        Parameters
        ----------
        params : dict
            Model parameters.
        state : dict
            Haiku state.
        rng_key : jax.random.PRNGKey
            PRNG key for stochastic layers.
        batch : tuple
            Training batch containing inputs and targets.
        is_training : bool
            Whether to run in training mode.

        Returns
        -------
        jnp.ndarray
            Scalar loss value.
        """
        pred, _ = net.apply(params, state, rng_key, batch, is_training)
        target = batch[1]
        # mask = batch[2]
        se = 0.5 * (pred - target) ** 2

        # nonzero = jnp.sum(mask, axis=0)
        # mse = (jnp.sum(jnp.where(mask, se, jnp.zeros_like(se)), axis=0) / nonzero).sum()
        mse = jnp.average(se)

        # Regularization (smoothness)
        first_diff = jnp.diff(pred, axis=1)
        first_diff_n = (first_diff - jnp.mean(first_diff, axis=1)[:, np.newaxis]) / jnp.std(
            first_diff, axis=1
        )[:, np.newaxis]

        first_diff_n = jnp.where(
            jnp.isfinite(first_diff_n), first_diff_n, jnp.zeros_like(first_diff_n)
        )
        roughness = ((jnp.diff(first_diff_n, axis=1) ** 2) / 4).sum()

        roughness_weight = 0

        return 1 / (roughness_weight + 1) * (mse + roughness_weight * roughness)

    @functools.partial(jax.jit, static_argnums=[5])
    def get_updates(params, state, rng_key, opt_state, batch, is_training):
        """Learning rule (stochastic gradient descent).

        Parameters
        ----------
        params : dict
            Model parameters.
        state : dict
            Haiku state.
        rng_key : jax.random.PRNGKey
            PRNG key.
        opt_state : optax.OptState
            Optimizer state.
        batch : tuple
            Training batch.
        is_training : bool
            Whether the update is for training.

        Returns
        -------
        tuple
            ``(loss, new_params, new_opt_state)``.
        """
        loss_val, grads = jax.value_and_grad(loss)(
            params, state, rng_key, batch, is_training=is_training
        )
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss_val, new_params, opt_state

    @jax.jit
    def ema_update(params, avg_params):
        """Update EMA parameters from current parameters."""
        return optax.incremental_update(params, avg_params, step_size=0.001)

    for epoch in range(conf["epochs"]):
        # Train/eval loop.
        train_loss = 0
        for train in train_loader:
            rng_key = next(key)
            loss_val, params, opt_state = get_updates(
                params, state, rng_key, opt_state, train, is_training=True
            )
            avg_params = ema_update(params, avg_params)

            train_loss += loss_val * len(train[0])
        train_loss /= len(train_data)

        test_loss = 0
        for test in test_loader:
            test_loss += loss(avg_params, state, None, test, is_training=False) * len(test[0])
        test_loss /= len(test_data)

        if writer is not None:
            train_loss, test_loss, lr = jax.device_get(
                (train_loss, test_loss, schedule(opt_state[1].count))
            )
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("LR", lr, epoch)

    @jax.jit
    def net_eval_fn(x):
        """JIT-compiled evaluation using averaged parameters.

        Parameters
        ----------
        x : array-like
            Input array for evaluation.

        Returns
        -------
        array-like
            Model outputs.
        """
        return net.apply(avg_params, state, None, x, is_training=False)[0]

    if writer is not None:
        test_loss = 0
        for test in test_loader:
            test_loss += loss(avg_params, state, None, test, is_training=False) * len(test[0])
        test_loss /= len(test_data)

        hparam_dict = dict(conf)
        if "final_activations" in hparam_dict:
            del hparam_dict["final_activations"]
        writer.add_hparams(hparam_dict, {"hparam/test_loss": np.asarray(test_loss)})
        writer.flush()
        writer.close()

    return net_eval_fn, avg_params, state
