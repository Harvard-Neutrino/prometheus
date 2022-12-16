import awkward as ak

def serialize_to_awkward(
    det,
    particles,
):

    # Only create array if any particles made light
    if not any([len(p.hits) > 0 for p in particles]):
        return None

    hit_functions = [
        ("string_id", lambda particles: [[h[0] for h in p.hits] for p in particles]),
        ("sensor_id", lambda particles: [[h[1] for h in p.hits] for p in particles]),
        ("t", lambda particles: [[h[2] for h in p.hits] for p in particles]),
        ("sensor_pos_x", lambda particles: [[det[(h[0], h[1])].pos[0] for h in p.hits] for p in particles]),
        ("sensor_pos_y", lambda particles: [[det[(h[0], h[1])].pos[1] for h in p.hits] for p in particles]),
        ("sensor_pos_z", lambda particles: [[det[(h[0], h[1])].pos[2] for h in p.hits] for p in particles]),
    ]

    outarr = ak.Array({k:f(particles) for k, f in hit_functions})

    return outarr
