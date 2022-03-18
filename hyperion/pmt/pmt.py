import scipy.stats
import numpy as np
from scipy.interpolate import UnivariateSpline


class SPETemplate:
    def __init__(self):
        self.components = [
            scipy.stats.expon(scale=1),
            scipy.stats.truncnorm(-1 / 0.3, 10, loc=1, scale=0.3),
        ]

        self.weights = [0.3, 0.7]

    def rvs(self, size, rng):
        pe = np.ones(size) * (-1.0)
        comp = rng.choice([0, 1], p=self.weights, size=size)

        is_comp_0 = comp == 0
        pe[is_comp_0] = self.components[0].rvs(size=is_comp_0.sum(), random_state=rng)

        is_comp_1 = comp == 1
        pe[is_comp_1] = self.components[1].rvs(size=is_comp_1.sum(), random_state=rng)
        return pe

    def pdf(self, xs):
        return self.weights[0] * self.components[0].pdf(xs) + self.weights[
            1
        ] * self.components[1].pdf(xs)


class PulseTemplate:
    def __init__(self):
        pass

    def __call__(self, xs, times, charges):
        return charges * scipy.stats.gumbel_r.pdf(
            xs[:, np.newaxis], loc=times + 2, scale=2
        )


def make_waveform(
    hits, spe_template, pulse_template, times=None, rng=np.random.RandomState(0)
):
    # jitter = scipy.stats.norm(0, 2, size=len(hits))
    # times = hits + jitter.rvs(size=len(hits))

    charges = spe_template.rvs(len(hits), rng=rng)

    if times is None:
        tmin = hits[0] - 6
        tmax = hits[-1] + 6

        times = np.arange(tmin, tmax, 2)

    wv = pulse_template(times, hits, charges).sum(axis=1)

    return wv, charges, times


def make_calc_wl_acceptance_weight(path_to_acc_data):
    wl_acc = np.loadtxt(path_to_acc_data, delimiter=",")

    zero_mask = wl_acc[:, 1] > 0
    safe_range = (np.min(wl_acc[zero_mask, 0]), np.max(wl_acc[zero_mask, 0]))

    wl_acc_spl = UnivariateSpline(
        wl_acc[zero_mask, 0], np.log(wl_acc[zero_mask, 1]), s=0.1
    )
    wlw_peak = scipy.optimize.minimize(lambda wl: -wl_acc_spl(wl), x0=[400]).x

    val_at_peak = np.exp(wl_acc_spl(wlw_peak))
    wl_acc_spl = UnivariateSpline(
        wl_acc[zero_mask, 0], np.log(wl_acc[zero_mask, 1] / val_at_peak), s=0.08
    )

    def calc_wl_acc(wavelength, peak_qe):
        if np.any((wavelength < safe_range[0]) | (wavelength > safe_range[1])):
            raise ValueError("Wavelength outside of safe range")
        return np.exp(wl_acc_spl(wavelength)) * peak_qe

    return calc_wl_acc
