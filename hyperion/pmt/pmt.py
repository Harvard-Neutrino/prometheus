import numpy as np
import scipy.stats
from scipy.interpolate import UnivariateSpline


class SPETemplate:
    """Single-photoelectron (SPE) template mixture model.

    Attributes
    ----------
    components : list
        List of ``scipy.stats`` distribution components for SPE shapes.
    weights : list
        Mixing weights for each component.
    """

    def __init__(self):
        """Initialize default SPE mixture components and weights.

        The default components are a combination of an exponential and a
        truncated normal to mimic a simple SPE shape.
        """
        self.components = [
            scipy.stats.expon(scale=1),
            scipy.stats.truncnorm(-1 / 0.3, 10, loc=1, scale=0.3),
        ]

        self.weights = [0.3, 0.7]

    def rvs(self, size, rng):
        """Draw random SPE charges from the mixture model.

        Parameters
        ----------
        size : int
            Number of samples to draw.
        rng : numpy.random.RandomState or numpy.random.Generator
            Random number generator used for sampling.

        Returns
        -------
        np.ndarray
            Array of sampled SPE charges.
        """
        pe = np.ones(size) * (-1.0)
        comp = rng.choice([0, 1], p=self.weights, size=size)

        is_comp_0 = comp == 0
        pe[is_comp_0] = self.components[0].rvs(size=is_comp_0.sum(), random_state=rng)

        is_comp_1 = comp == 1
        pe[is_comp_1] = self.components[1].rvs(size=is_comp_1.sum(), random_state=rng)
        return pe

    def pdf(self, xs):
        """Evaluate the SPE mixture probability density at input locations.

        Parameters
        ----------
        xs : array-like
            Points at which to evaluate the PDF.

        Returns
        -------
        np.ndarray
            PDF values corresponding to ``xs``.
        """
        return self.weights[0] * self.components[0].pdf(xs) + self.weights[1] * self.components[
            1
        ].pdf(xs)


class PulseTemplate:
    """Callable pulse template that generates per-hit pulse waveforms.

    The instance is callable with signature ``(xs, times, charges)`` and returns
    an array of per-time-bin pulse amplitudes formed by summing a simple Gumbel
    pulse shape for each hit time and charge.
    """

    def __init__(self):
        """Create a callable pulse template.

        The instance can be called to produce per-hit pulse shapes.
        """
        pass

    def __call__(self, xs, times, charges):
        """Generate per-hit pulse waveforms and scale by charges.

        Parameters
        ----------
        xs : np.ndarray
            Time grid (1-D) for the waveform.
        times : array-like
            Hit times.
        charges : array-like
            Per-hit charges.

        Returns
        -------
        np.ndarray
            Per-time-bin pulse amplitudes with shape ``(len(xs), len(times))``.
        """
        return charges * scipy.stats.gumbel_r.pdf(xs[:, np.newaxis], loc=times + 2, scale=2)


def make_waveform(hits, spe_template, pulse_template, times=None, rng=np.random.RandomState(0)):
    """Generate a waveform from hits using SPE and pulse templates.

    Parameters
    ----------
    hits : array-like
        Hit times (sorted).
    spe_template : SPETemplate
        Single-photoelectron template with ``rvs`` and ``pdf`` methods.
    pulse_template : callable
        Callable with signature ``(xs, times, charges)`` returning per-hit waveforms.
    times : array-like, optional
        Time grid for the waveform. If None, inferred from hits.
    rng : numpy.random.RandomState, optional
        Random number generator used for sampling SPE charges.

    Returns
    -------
    tuple
        ``(wv, charges, times)`` where ``wv`` is the waveform array, ``charges`` are
        sampled SPE charges, and ``times`` is the time grid.
    """

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
    """Build a wavelength-acceptance function from CSV data.

    Parameters
    ----------
    path_to_acc_data : str
        Path to CSV file containing wavelength and acceptance values.

    Returns
    -------
    callable
        Function ``(wavelength, peak_qe) -> acceptance_weight`` that interpolates
        the acceptance and scales by ``peak_qe``.
    """

    wl_acc = np.loadtxt(path_to_acc_data, delimiter=",")

    zero_mask = wl_acc[:, 1] > 0
    safe_range = (np.min(wl_acc[zero_mask, 0]), np.max(wl_acc[zero_mask, 0]))

    wl_acc_spl = UnivariateSpline(wl_acc[zero_mask, 0], np.log(wl_acc[zero_mask, 1]), s=0.1)
    wlw_peak = scipy.optimize.minimize(lambda wl: -wl_acc_spl(wl), x0=[400]).x

    val_at_peak = np.exp(wl_acc_spl(wlw_peak))
    wl_acc_spl = UnivariateSpline(
        wl_acc[zero_mask, 0], np.log(wl_acc[zero_mask, 1] / val_at_peak), s=0.08
    )

    def calc_wl_acc(wavelength, peak_qe):
        """Interpolate and scale wavelength acceptance.

        Parameters
        ----------
        wavelength : float or array-like
            Wavelength(s) in nanometres.
        peak_qe : float
            Peak quantum efficiency scaling factor.

        Returns
        -------
        float or array-like
            Acceptance weight(s) for the provided wavelength(s).
        """
        if np.any((wavelength < safe_range[0]) | (wavelength > safe_range[1])):
            raise ValueError("Wavelength outside of safe range")
        return np.exp(wl_acc_spl(wavelength)) * peak_qe

    return calc_wl_acc
