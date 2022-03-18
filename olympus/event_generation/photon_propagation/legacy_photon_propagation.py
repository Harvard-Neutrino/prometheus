"""Implements photon propagation."""
import awkward as ak
import numpy as np
from hyperion.models.photon_arrival_time.pdf import sample_exp_exp_exp
from .constants import Constants

import jax.numpy as jnp
from jax import random, vmap, jit


exp_exp_exp_sampler = jit(sample_exp_exp_exp)


@njit(nopython=True)
def sample_times(pdf_params, sources, module_coords, module_efficiencies, time_geo):

    all_times_det = []
    for idom in range(module_coords.shape[0]):

        this_times = []
        total_length = 0
        for isource in range(len(sources)):
            pars = pdf_params[isource, idom]
            usf = 1 - 10 ** (-pars[5])
            surv_ratio = 10 ** pars[6]
            n_ph_tot = np.random.poisson(
                surv_ratio * sources[isource].amp * module_efficiencies[idom]
            )

            n_direct, n_indirect = np.random.multinomial(n_ph_tot, [usf, 1 - usf])

            all_samples = np.empty(n_ph_tot)

            """
            Can't use this with numba yet
            expon_samples = sampler(*pars[:-2], size=n_indirect, rstate=rstate) + 2
            """
            expon_samples = (
                exp_exp_exp_sampler(
                    pars[0], pars[1], pars[2], pars[3], pars[4], n_indirect
                )
                + 2
            )
            uni_samples = np.random.uniform(0, 2, size=n_direct)

            all_samples[:n_direct] = uni_samples
            all_samples[n_direct:] = expon_samples

            all_samples += time_geo[isource, idom]
            this_times.append(all_samples)
            total_length += n_ph_tot

        this_times_arr = np.empty(total_length)
        i = 0
        for tt in this_times:
            this_times_arr[i : i + tt.shape[0]] = tt  # noqa: E203
            i += tt.shape[0]

        all_times_det.append(this_times_arr)

    return all_times_det


def make_generate_photons_nn(model_func):
    """
    Build an arival time sampling function.

    This function uses a pytorch model to predict the pdf parameters
    of a triple-exponential mixture model fitted to the arrival time distributions.
    """

    def generate_photons_nn(
        module_coords,
        module_efficiencies,
        sources,
        seed=31337,
        c_vac=Constants.c_vac,
        n_gr=Constants.n_gr,
    ):

        all_times_det = []
        np.random.seed(seed)

        inp_pars, time_geo = source_to_model_input(module_coords, sources, c_vac, n_gr)

        pdf_params = (model_func(inp_pars).cpu().detach().numpy()).reshape(
            [len(sources), module_coords.shape[0], 9]
        )[..., :7]

        all_times_det = sample_times(
            pdf_params, sources, module_coords, module_efficiencies, time_geo
        )

        all_times_det = ak.sort(ak.Array(all_times_det))
        return all_times_det

    return generate_photons_nn


vpoisson = jit(vmap(random.poisson, in_axes=0))


def interpolate_hist(hist, x, x_eval):
    dx = jnp.diff(x)
    dy = jnp.diff(hist)

    x_idx = jnp.searchsorted(x, x_eval, side="right") - 1
    fractional = x_eval - x[x_idx]

    interpolated = jnp.where(
        (x_idx >= 0) & (x_idx < x.shape[0] - 2),
        hist[x_idx] + dy[x_idx] / dx[x_idx] * fractional,
        jnp.nan,
    )
    return interpolated


def make_generate_bin_amplitudes_nn(
    det, model_func, binning, c_medium_f, prediction=False
):
    """
    Build a binned arival time sampling function.

    This function uses a pytorch model to predict the bin contents of the arrival time distribution.

    Parameters:
        model_path: str
            Path to pytorch model
        prediction: bool
            If True, return predicted amplitudes instead of poisson samples
    """

    nbins = len(binning) - 1

    binning = binning
    bin_width = binning[1] - binning[0]

    module_coords = jnp.asarray(det.module_coords)
    module_efficiencies = jnp.asarray(det.module_efficiencies)

    def generate_samples(tmin, tmax, time_geo, pred_frac_log, source_amps):
        new_binning = jnp.arange(tmin, tmax + bin_width, bin_width)
        new_binning_offsets_bws = (time_geo + binning[0] - tmin) / bin_width

        new_binning_offsets = jnp.int32(jnp.floor(new_binning_offsets_bws))
        new_binning_offsets_frac = new_binning_offsets_bws - new_binning_offsets
        interpolated = (
            pred_frac_log[:, :-1]
            + jnp.diff(pred_frac_log[:, :], axis=1)
            / bin_width
            * new_binning_offsets_frac[..., np.newaxis]
        )

        interpolated_amps = jnp.exp(interpolated) * source_amps[:, np.newaxis]

        this_pred_amplitudes = jnp.zeros(new_binning.shape[0] - 1)

        for isource in range(len(sources)):
            blower = new_binning_offsets[isource]
            bupper = new_binning_offsets[isource] + nbins - 1
            sli = slice(blower, bupper)
            this_pred_amplitudes.at[sli].add(interpolated_amps[isource])

        if prediction:
            samples.append((this_pred_amplitudes, new_binning))
        else:
            key, subkey = random.split(rngkey)
            pkeys = random.split(subkey, this_pred_amplitudes.shape[0])

            samples.append(
                (
                    vpoisson(
                        pkeys,
                        this_pred_amplitudes,
                    ),
                    new_binning,
                )
            )

    def generate(sources, rng_key):
        rng_key, subkey = random.split(rng_key)

        source_pos = jnp.empty((len(sources), 3))
        source_dir = jnp.empty((len(sources), 3))
        source_t0 = jnp.empty(len(sources))
        source_amp = jnp.empty(len(sources))

        for i in range(len(sources)):
            source_pos.at[i].set(sources[i].pos)
            source_dir.at[i].set(sources[i].dir)
            source_t0.at[i].set(sources[i].t0)
            source_amp.at[i].set(sources[i].amp)

        inp_pars, time_geo = sources_to_model_input(
            module_coords, source_pos, source_dir, source_t0, c_medium_f(700)
        )

        inp_pars = inp_pars.reshape(len(sources) * module_coords.shape[0], 2)

        pred_frac_log = model_func(inp_pars).reshape(
            [len(sources), module_coords.shape[0], nbins]
        )

        tmin = jnp.min(time_geo, axis=0) + binning[0]
        tmax = jnp.max(time_geo, axis=0) + binning[-1]

    def generate_bin_amp_nn_per_source(
        source_amp,
        source_pos,
        source_dir,
        source_t0,
        rng_key,
    ):

        rng_key, subkey = random.split(rng_key)

        inp_pars, time_geo = source_to_model_input(
            module_coords, source_pos, source_dir, source_t0, c_medium_f(700)
        )

        pred_frac_log = model_func(inp_pars).reshape([module_coords.shape[0], nbins])

        # We have the predicted amplitudes relative to each source's time.
        # Now construct a binning that covers the full range spanned by all light sources.

        samples = []

        for imod in range(module_coords.shape[0]):

            new_binning = jnp.arange(tmin[imod], tmax[imod] + bin_width, bin_width)
            new_binning_offsets_bws = (
                time_geo[:, imod] + binning[0] - tmin[imod]
            ) / bin_width

            new_binning_offsets = jnp.int32(jnp.floor(new_binning_offsets_bws))
            new_binning_offsets_frac = new_binning_offsets_bws - new_binning_offsets
            # print(new_binning_offsets, new_binning_offsets_frac)
            interpolated = (
                pred_frac_log[:, imod, :-1]
                + jnp.diff(pred_frac_log[:, imod, :], axis=1)
                / bin_width
                * new_binning_offsets_frac[..., np.newaxis]
            )

            interpolated_amps = jnp.exp(interpolated) * source_amps[:, np.newaxis]

            this_pred_amplitudes = jnp.zeros(new_binning.shape[0] - 1)

            for isource in range(len(sources)):
                blower = new_binning_offsets[isource]
                bupper = new_binning_offsets[isource] + nbins - 1
                sli = slice(blower, bupper)
                this_pred_amplitudes.at[sli].add(interpolated_amps[isource])

            if prediction:
                samples.append((this_pred_amplitudes, new_binning))
            else:
                key, subkey = random.split(rngkey)
                pkeys = random.split(subkey, this_pred_amplitudes.shape[0])

                samples.append(
                    (
                        vpoisson(
                            pkeys,
                            this_pred_amplitudes,
                        ),
                        new_binning,
                    )
                )

        return (samples, time_geo)

    return generate_bin_amp_nn
