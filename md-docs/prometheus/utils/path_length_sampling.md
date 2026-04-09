Module prometheus.utils.path_length_sampling
============================================

Functions
---------

    
`path_length_sampling(E: float, pdg_id: int) ‑> float`
:   quick and dirty path length sampling
    
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