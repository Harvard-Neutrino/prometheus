# path_length_sampling.py
# Authors: Stephan Meighen-Berger
# Quick and dirty sampling function for injected hadrons (their offset from the interaction vertex if not given)

import numpy as np


def path_length_sampling(E: float, pdg_id: int) -> float:
    """ quick and dirty path length sampling

    Parameters
    ----------
    E: float/np.array
        The energy of the particle(s)
    pdg_id: int
        The pdg id of the particle

    Returns
    -------
    dist: float/np.array
        The sampled travelled distance in cm with the same
        shape as E
    """
    # TODO: Add energy dependence
    dens_water = 0.918  # in g/cm^-3
    if pdg_id in (2212, 2112):
        lamd = 83.2 / dens_water  # in cm
    elif pdg_id in (211, 111, -211, 130, 310, 311, 321, -321):
        lamd = 115.2 / dens_water  # in cm
    else:
        print("Unknown particle")
        print("Assuming it travels like a pion")
        lamd = 115.2 / dens_water
    if type(E) is np.ndarray:
        return np.random.exponential(scale=lamd, size=E.shape)
    else:
        return np.random.exponential(scale=lamd, size=None)