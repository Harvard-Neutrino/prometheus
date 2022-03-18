import distrax
import jax.numpy as jnp
import haiku as hk
import optax
import jax


def make_conditioner(hidden_sizes, out_params_activ, init_zero=True):
    """
    Build a conditioner MLP.

    Parameters:
        hidden_sizes: List
            List of layer sizes
        out_params_traf: List
            List of activations applied to final layer.
            Can be None, in which case no transformation is applied.
    """

    def final_activation(x):
        for i, op_t in enumerate(out_params_activ):
            if op_t is not None:
                x = x.at[:, i].set(op_t(x[:, i]))
        return x

    n_out_params = len(out_params_activ)

    if init_zero:
        final_linear = hk.Linear(n_out_params, w_init=jnp.zeros, b_init=jnp.zeros)
    else:
        final_linear = hk.Linear(n_out_params)

    return hk.Sequential(
        [
            hk.Flatten(preserve_dims=1),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            final_linear,
            final_activation,
        ]
    )


def make_shape_conditioner_fn(
    mlp_hidden_size, mlp_num_layers, flow_num_bins, flow_num_layers
):

    num_bijector_params = 3 * flow_num_bins + 1
    num_bij_params = num_bijector_params * flow_num_layers

    @hk.without_apply_rng
    @hk.transform
    def shape_conditioner_fn(x):
        return make_conditioner(
            [mlp_hidden_size] * mlp_num_layers, [None] * num_bij_params
        )(x)

    return shape_conditioner_fn


def make_spl_flow(spl_params, rmin, rmax):
    """
    Make multiple spline flows.

    Paramaters:
        spl_params: List
            List of spline parameters per layer
        rmin, rmax: float
            Min and max range for spline
    """
    layers = []
    for spl_p in spl_params:
        layers.append(
            distrax.RationalQuadraticSpline(
                spl_p,
                range_min=rmin,
                range_max=rmax,
            )
        )
    return layers


def traf_dist_builder(flow_num_layers, flow_range, return_base=False):
    def make_transformed_dist(traf_params):
        spl_params = jnp.split(traf_params, flow_num_layers, axis=-1)

        base_dist = distrax.Gamma(1.5, 1 / 10)
        flow = make_spl_flow(spl_params, flow_range[0], flow_range[1]) + [
            distrax.ScalarAffine(shift=4, scale=1)
        ]
        flow = distrax.Inverse(distrax.Chain(flow))
        # flow = distrax.Chain(flow)
        transformed = distrax.Transformed(base_dist, flow)

        if return_base:
            return base_dist, flow

        return transformed

    return make_transformed_dist


def eval_log_prob(dist_builder, traf_params, samples):
    dist = dist_builder(traf_params)
    return dist.log_prob(samples)


def sample_shape_model(dist_builder, traf_params, n_photons, seed):
    base_dist, trafo = dist_builder(traf_params)
    base_samples = base_dist.sample(seed=seed, sample_shape=n_photons)
    return trafo.forward(base_samples)


def train_shape_model(config, train_loader, test_loader, seed=1337, writer=None):

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
        return optax.incremental_update(params, avg_params, step_size=0.001)

    @jax.jit
    def loss_fn(params, cond, samples):

        traf_params = shape_conditioner.apply(params, cond)

        lprobs = eval_log_prob(dist_builder, traf_params, samples)
        finite = jnp.isfinite(lprobs)
        loss = -jnp.mean(lprobs * finite)
        return loss

    @jax.jit
    def update(params, opt_state, cond, samples):
        """Single SGD update step."""
        lval, grads = jax.value_and_grad(loss_fn)(params, cond, samples)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, lval

    scheduler = optax.cosine_decay_schedule(config["lr"], config["steps"], alpha=0.0)

    # scheduler = lambda _ : config["lr"]
    optimizer = optax.adam(learning_rate=scheduler)

    prng_seq = hk.PRNGSequence(seed)

    params = avg_params = shape_conditioner.init(next(prng_seq), jnp.ones((1, 2)))
    opt_state = optimizer.init(params)

    log_every = 100

    train_iter = iter(train_loader)
    for i in range(1, config["steps"] + 1):

        train = next(train_iter)
        cond = jnp.concatenate(train[:2]).T
        samples = jnp.squeeze(train[2])

        params, opt_state, train_loss = update(params, opt_state, cond, samples)
        avg_params = ema_update(params, avg_params)

        if (i % log_every) == 0:

            test_loss = 0
            for test in test_loader:
                cond = jnp.concatenate(test[:2]).T
                samples = jnp.squeeze(test[2])
                val_loss = loss_fn(avg_params, cond, samples)
                test_loss += val_loss
            test_loss /= test_loader._n_batches

            train_loss, test_loss = jax.device_get(
                (
                    train_loss,
                    test_loss,
                )
            )

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, i)
                writer.add_scalar("Loss/test", test_loss, i)
                # writer.add_scalar("LR", lr, epoch)
                writer.flush()
            print(f"Epoch: {i} \t Train/Test: {train_loss:.3E} / {test_loss:.3E}")
    return params


def make_counts_net_fn(config):
    @hk.without_apply_rng
    @hk.transform
    def net_fn(x):
        net = make_conditioner(
            [config["mlp_hidden_size"]] * config["mlp_num_layers"],
            [None],
            init_zero=False,
        )
        return net(x)

    return net_fn


def train_counts_model(config, train_loader, test_loader, seed=1337, writer=None):

    net_fn = make_counts_net_fn(config)

    @jax.jit
    def loss_fn(params, batch):
        inp = jnp.concatenate(batch[:2]).T
        out = net_fn.apply(params, inp).squeeze()
        return 0.5 * jnp.average((out - batch[2]) ** 2)

    @jax.jit
    def update(params, opt_state, batch):
        """Single SGD update step."""
        lval, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, lval

    scheduler = optax.cosine_decay_schedule(config["lr"], config["steps"], alpha=0.0)

    optimizer = optax.adam(learning_rate=scheduler)

    prng_seq = hk.PRNGSequence(42)

    params = net_fn.init(next(prng_seq), jnp.ones((1, 2)))
    opt_state = optimizer.init(params)

    log_every = 100

    train_iter = iter(train_loader)
    for i in range(1, config["steps"] + 1):

        train = next(train_iter)

        params, opt_state, train_loss = update(params, opt_state, train)

        if (i % log_every) == 0:

            test_loss = 0
            for test in test_loader:
                val_loss = loss_fn(params, test)
                test_loss += val_loss
            test_loss /= test_loader._n_batches

            train_loss, test_loss = jax.device_get(
                (
                    train_loss,
                    test_loss,
                )
            )

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, i)
                writer.add_scalar("Loss/test", test_loss, i)
                # writer.add_scalar("LR", lr, epoch)
                writer.flush()
            print(f"Epoch: {i} \t Train/Test: {train_loss:.3E} / {test_loss:.3E}")

    if writer is not None:
        test_loss = 0
        test_loss = 0
        for test in test_loader:
            val_loss = loss_fn(params, test)
            test_loss += val_loss
        test_loss /= test_loader._n_batches
        test_loss = jax.device_get(test_loss)

        hparam_dict = dict(config)
        writer.add_hparams(hparam_dict, {"hparam/test_loss": test_loss})
        writer.flush()
        writer.close()
    return params
