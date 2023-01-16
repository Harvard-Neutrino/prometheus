import awkward as ak
import numpy as np

def serialize_particles_to_awkward(
    det,
    particles,
):

    # Only create array if any particles made light
    if not any([len(p.hits) > 0 for p in particles]):
        return None

    xyz = [
        np.transpose(np.array([det[(h.string_id, h.om_id)].pos for h in p.hits])) for p in particles
    ]
    outdict = {}
    for idx, var in enumerate("x y z".split()):
        outdict[f"sensor_pos_{var}"] = [x[idx] if x.shape[0] > 0 else np.array([]) for x in xyz]

    hit_functions = [
        ("string_id", lambda particles: [[h.string_id for h in p.hits] for p in particles]),
        ("sensor_id", lambda particles: [[h.om_id for h in p.hits] for p in particles]),
        ("t", lambda particles: [[h.time for h in p.hits] for p in particles]),
        #("sensor_pos_x", lambda particles: [[det[(h.string_id, h.om_id)].pos[0] for h in p.hits] for p in particles]),
        #("sensor_pos_y", lambda particles: [[det[(h.string_id, h.om_id)].pos[1] for h in p.hits] for p in particles]),
        #("sensor_pos_z", lambda particles: [[det[(h.string_id, h.om_id)].pos[2] for h in p.hits] for p in particles]),
    ]

    for field, fxn in hit_functions:
        outdict[field] = fxn(particles)
    outarr = ak.Array(outdict)
    #outarr = ak.Array({k:f(particles) for k, f in hit_functions})

    return outarr
