import awkward as ak
import numpy as np

from .accumulate_hits import accumulate_hits
from prometheus.detector import Detector
from prometheus.injection.injection import Injection

def serialize_particles_to_awkward(
    det: Detector,
    injection: Injection
):
    all_hits = []
    for injection_event in injection:
        all_hits.append(accumulate_hits(injection_event.final_states))

    if not any([len(h) > 0 for h in all_hits]):
        return None
    xyz = [
        np.array([det[(h.string_id, h.om_id)].pos for h, _ in event_hits]).transpose()
        for event_hits in all_hits
    ]

    outdict = {}
    for idx, var in enumerate("x y z".split()):
        outdict[f"sensor_pos_{var}"] = [
            x[idx] if x.shape[0] > 0 else np.array([]) for x in xyz
        ]

    hit_functions = [
        ("string_id", lambda x: [[h.string_id for h, _ in event_hits] for event_hits in x]),
        ("sensor_id", lambda x: [[h.om_id for h, _ in event_hits] for event_hits in x]),
        ("t", lambda x: [[h.time for h, _ in event_hits] for event_hits in x]),
        ("id_idx", lambda x: [[id_str for _, id_str in event_hits] for event_hits in x]),
    ]

    for field, fxn in hit_functions:
        outdict[field] = fxn(all_hits)
    outarr = ak.Array(outdict)
    return outarr
