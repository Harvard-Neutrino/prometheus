import numpy as np

# Find center of gravity of an event
def find_cog(event_dict, detector):
    keys = event_dict.keys()
    charges = np.array([len(event_dict[key]) for key in keys])
    xyz = np.array([detector[key].pos for key in keys])
    cog = np.sum(xyz.T * charges, axis=1) / np.sum(charges)
    return cog
