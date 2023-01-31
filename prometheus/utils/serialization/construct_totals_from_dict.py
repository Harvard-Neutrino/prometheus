import numpy as np
import awkward as ak

def construct_totals_from_dict(
        det,
        fill_dict
    ):
    particle_keys = [
        k for k in fill_dict.keys()
        if k not in "event_id mc_truth".split()
    ]
    sensor_id_all = np.array(
        [np.array([], dtype=np.int64) for _ in range(len(fill_dict[particle_keys[0]]["sensor_id"]))]
    )
    t_all = np.array(
        [np.array([]) for _ in range(len(fill_dict[particle_keys[0]]["t"]))]
    )
    for i, k in enumerate(particle_keys):
        if i==0:
            cur_t = fill_dict[k]["t"]
            cur_sensor_id = fill_dict[k]["sensor_id"]
        else:
            cur_t = [
                x if np.all(x!=-1) else [] for x in fill_dict[k]["t"]
            ]
            cur_sensor_id = [
                x if np.all(x!=-1) else [] for x in fill_dict[k]["sensor_id"]
            ]
        t_all = ak.concatenate(
            (t_all, cur_t),
            axis=1
        )
        sensor_id_all = ak.concatenate(
            (sensor_id_all, cur_sensor_id),
            axis=1
        )
    sensor_pos_all = np.array(
        [
            det.module_coords[hits]
            for hits in sensor_id_all
        ],
        dtype=object
    )
    sensor_string_id_all = np.array(
        [
            np.array(det._om_keys)[event]
            for event in sensor_id_all
        ],
        dtype=object
    )
    fill_dict['total'] = {
        'sensor_id': sensor_id_all,
        'sensor_pos_x': ak.Array([
            event[:, 0] for event in sensor_pos_all
        ]),
        'sensor_pos_y': ak.Array([
            event[:, 1] for event in sensor_pos_all
        ]),
        'sensor_pos_z': ak.Array([
            event[:, 2] for event in sensor_pos_all
        ]),
        'string_id': ak.Array([
            event[:, 0] for event in sensor_string_id_all
        ]),
        't':t_all,
    }

