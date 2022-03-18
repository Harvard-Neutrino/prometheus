import numpy as np
import scipy.stats
from jax import numpy as jnp
from jax import value_and_grad, jit
from sphere.distribution import kent_me, fb8


def expon_pdf(x, a):
    """Exponential PDF."""
    return 1 / a * jnp.exp(-x / a)


def make_exp_exp(data, weights):
    """
    Create a two-exponential mixture model pdf.

    This functions returns the likelihood evaluated on data and weights, and
    the likelihood function.

    Parameters:
        data: ndarray
        weights: ndarray

    """

    def func(xs, scale1, scale2, mix):
        lower = jnp.min(jnp.array([scale1, scale2])) * 100
        upper = jnp.max(jnp.array([scale1, scale2])) * 100

        res = jnp.log(mix * expon_pdf(xs, lower) + (1 - mix) * expon_pdf(xs, upper))

        return res

    def obj(scale1, scale2, mix):
        val = -jnp.sum((func(data, scale1, scale2, mix) * weights))
        return val

    return jit(value_and_grad(obj, [0, 1, 2])), func


def make_obj_func(pdf, data, weights, nargs):
    """
    Build a likelihood from a PDF, data points and weights.

    Returns:
    Objective function and gradient
    """

    def obj(*pars):
        val = -jnp.sum((pdf(data, *pars) * weights))
        return val

    return jit(value_and_grad(obj, list(range(nargs))))


def sample_exp_exp_exp(scale1, scale2, scale3, w1, w2, size):
    """
    Sampling function. To ensure numba compatibility set random state before
    """
    scales = np.array([scale1, scale2, scale3]) * 100
    scales = np.sort(scales)

    weights = (
        np.array([np.sin(w1) * np.cos(w2), np.sin(w1) * np.sin(w2), np.cos(w1)]) ** 2
    )

    n_per_comp = np.random.multinomial(size, weights)

    samples = np.empty(size)

    npc_cum = np.cumsum(n_per_comp)
    uni = np.random.uniform(0, 1, size=size)

    samples[: npc_cum[0]] = -np.log(uni[: npc_cum[0]]) * scales[0]
    samples[npc_cum[0] : npc_cum[1]] = -np.log(uni[npc_cum[0] : npc_cum[1]]) * scales[1]
    samples[npc_cum[1] :] = -np.log(uni[npc_cum[1] :]) * scales[2]

    return samples


def make_exp_exp_exp():
    """
    Create a three-exponential mixture model pdf.
    """

    def to_three_par(w1, w2):
        three = (
            jnp.array(
                [jnp.sin(w1) * jnp.cos(w2), jnp.sin(w1) * jnp.sin(w2), jnp.cos(w1)]
            )
            ** 2
        )
        return three

    def func(xs, scale1, scale2, scale3, w1, w2):

        scales = jnp.array([scale1, scale2, scale3]) * 100
        scales = jnp.sort(scales)

        weights = to_three_par(w1, w2)

        res = jnp.log(
            weights[0] * expon_pdf(xs, scales[0])
            + weights[1] * expon_pdf(xs, scales[1])
            + weights[2] * expon_pdf(xs, scales[2])
        )

        return res

    return func


def make_gamma_exponential(data, weights):
    """
    Create a gamma-exponential mixture model pdf.

    This functions returns the likelihood evaluated on data and weights, and
    the likelihood function.

    Parameters:
        data: ndarray
        weights: ndarray

    """

    def func(xs, args):
        a, scale, scale2, mix = args
        scale *= 100
        scale2 *= 100

        f1 = np.log(mix) + scipy.stats.gamma.logpdf(xs, a, scale=scale)
        f2 = np.log(1 - mix) + scipy.stats.expon.logpdf(xs, scale=scale2)
        stacked = np.vstack([f1, f2])

        res = scipy.special.logsumexp(stacked, axis=0)
        res[~np.isfinite(res)] = 1e4

        return res

    def obj(args):
        return -(func(data, args) * weights).sum()

    return obj, func


def fb5_mle(xs, weights, warning="warn"):
    """
    Fits FB5 distribution to weighted data.
    """

    # method that generates the minus L to be minimized
    # x = theta phi psi kappa beta eta alpha rho
    def minus_log_likelihood(x):
        if np.any(np.isnan(x)):
            return np.inf
        if x[3] < 0 or x[4] < 0:
            return np.inf

        return -(fb8(*x).log_pdf(xs) * weights).sum() / weights.sum()

    # first get estimated moments
    k_me = kent_me(xs)

    theta, phi, psi, kappa, beta = k_me.theta, k_me.phi, k_me.psi, k_me.kappa, k_me.beta

    # here the mle is done
    x_start = np.array([theta, phi, psi, kappa, beta])

    # First try a FB5 fit
    # constrain kappa, beta >= 0 and 2*beta <= kappa for FB5 (Kent 1982)

    cons = (
        {"type": "ineq", "fun": lambda x: x[3] - 2 * x[4]},
        {"type": "ineq", "fun": lambda x: x[3]},
        {"type": "ineq", "fun": lambda x: x[4]},
    )
    all_values = scipy.optimize.minimize(
        minus_log_likelihood,
        x_start,
        method="SLSQP",
        constraints=cons,
        options={"disp": False, "ftol": 1e-08, "maxiter": 100},
    )
    return all_values.x
