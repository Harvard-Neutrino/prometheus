import numpy as np

def find_cog(event_dict, detector):
    """Find the center of gravity of an event."""
    keys = event_dict.keys()
    charges = np.array([len(event_dict[key]) for key in keys])
    xyz = np.array([detector[key].pos for key in keys])
    cog = np.sum(xyz.T * charges, axis=1) / np.sum(charges)
    return cog
