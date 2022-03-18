import pickle
import functools
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hyperion.data import DataLoader


class HistMLP(hk.Module):
    def __init__(self, output_size, layers, dropout, final_activations, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.layers = layers
        self.dropout = dropout
        self.final_activations = final_activations

    def __call__(self, x, is_training):
        for n_per_layer in self.layers:
            x = hk.Linear(n_per_layer)(x)
            # x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training=is_training)
            x = jax.nn.relu(x)
            if is_training:
                key = hk.next_rng_key()
                x = hk.dropout(key, self.dropout, x)

        x = hk.Linear(self.output_size)(x)

        if self.final_activations is not None:
            x = self.final_activations(x)

        return x


def make_forward_fn(conf):
    layers = [conf["n_neurons"], conf["n_neurons"], conf["n_neurons"]]

    def forward_fn(batch, is_training):
        inp = jnp.asarray(batch[0], dtype=jnp.float32)
        return HistMLP(conf["n_out"], layers, conf["dropout"], None)(inp, is_training)

    return forward_fn


def make_eval_forward_fn(conf):
    layers = [conf["n_neurons"], conf["n_neurons"], conf["n_neurons"]]

    def forward_fn(inp):
        return HistMLP(conf["n_out"], layers, conf["dropout"], None)(inp, False)

    return forward_fn


def make_logp1_trafo(scale):
    def trafo(data):
        return np.log(data * scale + 1)

    def rev_trafo(data):
        return jnp.exp(data - 1) / scale

    return trafo, rev_trafo


def make_net_eval_from_pickle(path):
    (params, state, conf, binning, trafo_scale) = pickle.load(open(path, "rb"))
    forward_fn = make_eval_forward_fn(conf)
    net = hk.transform_with_state(forward_fn)

    _, rev_trafo = make_logp1_trafo(trafo_scale)

    @jax.jit
    def net_eval_fn(x):
        return rev_trafo(net.apply(params, state, None, x)[0])

    return net_eval_fn, binning


def train_net(conf, train_data, test_data, writer, rng):

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
        pred, _ = net.apply(params, state, rng_key, batch, is_training)
        target = batch[1]
        # mask = batch[2]
        se = 0.5 * (pred - target) ** 2

        # nonzero = jnp.sum(mask, axis=0)
        # mse = (jnp.sum(jnp.where(mask, se, jnp.zeros_like(se)), axis=0) / nonzero).sum()
        mse = jnp.average(se)

        # Regularization (smoothness)
        first_diff = jnp.diff(pred, axis=1)
        first_diff_n = (
            first_diff - jnp.mean(first_diff, axis=1)[:, np.newaxis]
        ) / jnp.std(first_diff, axis=1)[:, np.newaxis]

        first_diff_n = jnp.where(
            jnp.isfinite(first_diff_n), first_diff_n, jnp.zeros_like(first_diff_n)
        )
        roughness = ((jnp.diff(first_diff_n, axis=1) ** 2) / 4).sum()

        roughness_weight = 0

        return 1 / (roughness_weight + 1) * (mse + roughness_weight * roughness)

    @functools.partial(jax.jit, static_argnums=[5])
    def get_updates(params, state, rng_key, opt_state, batch, is_training):
        """Learning rule (stochastic gradient descent)."""
        l, grads = jax.value_and_grad(loss)(
            params, state, rng_key, batch, is_training=is_training
        )
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return l, new_params, opt_state

    @jax.jit
    def ema_update(params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)

    for epoch in range(conf["epochs"]):
        # Train/eval loop.
        train_loss = 0
        for train in train_loader:
            rng_key = next(key)
            l, params, opt_state = get_updates(
                params, state, rng_key, opt_state, train, is_training=True
            )
            avg_params = ema_update(params, avg_params)

            train_loss += l * len(train[0])
        train_loss /= len(train_data)

        test_loss = 0
        for test in test_loader:
            test_loss += loss(avg_params, state, None, test, is_training=False) * len(
                test[0]
            )
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
        return net.apply(avg_params, state, None, x, is_training=False)[0]

    if writer is not None:
        test_loss = 0
        for test in test_loader:
            test_loss += loss(avg_params, state, None, test, is_training=False) * len(
                test[0]
            )
        test_loss /= len(test_data)

        hparam_dict = dict(conf)
        if "final_activations" in hparam_dict:
            del hparam_dict["final_activations"]
        writer.add_hparams(hparam_dict, {"hparam/test_loss": np.asarray(test_loss)})
        writer.flush()
        writer.close()

    return net_eval_fn, avg_params, state
