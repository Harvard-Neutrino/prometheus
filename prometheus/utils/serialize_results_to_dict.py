import awkward as ak
import numpy as np

def serialize_results_to_dict(
    det,
    results,
    record,
):
    # TODO: Optimize this. Currently this is extremely inefficient.
        # first set
        all_ids_1 = []
        if not any([len(event) > 0 for event in results]):
            return
        xx = 0
        for event in results:
            xx += 1
            dom_ids_1 = []
            for dom_idx, dom in enumerate(event):
                if len(dom) > 0:
                    dom_ids_1.append([dom_idx] * len(dom))
            dom_ids_1 = ak.flatten(ak.Array(dom_ids_1), axis=None)
            all_ids_1.append(dom_ids_1)
        all_ids_1 = ak.Array(all_ids_1)
        all_hits_1 =  []
        for event in results:
            all_hits_1.append(ak.flatten(event, axis=None))
        all_hits_1 = ak.Array(all_hits_1)
        # Positional sensor information
        sensor_pos_1 = np.array([
            det.module_coords[hits]
            for hits in all_ids_1
        ], dtype=object)
        sensor_string_id_1 = np.array([
            np.array(det._om_keys)[event]
            for event in all_ids_1
        ], dtype=object)
        # The losses
        loss_counts = np.array([[
            source.n_photons[0] for source in event.sources
        ] for event in record], dtype=object)
        # This is as inefficient as possible
        d = {
            'sensor_id': all_ids_1,
            'sensor_pos_x': np.array([
                event[:, 0] for event in sensor_pos_1
            ], dtype=object),
            'sensor_pos_y': np.array([
                event[:, 1] for event in sensor_pos_1
            ], dtype=object),
            'sensor_pos_z': np.array([
                event[:, 2] for event in sensor_pos_1
            ], dtype=object),
            'string_id': np.array([
                event[:, 0] for event in sensor_string_id_1
            ], dtype=object),
            't': all_hits_1,
            'loss_pos_x': np.array([[
                source.position[0] for source in event.sources
            ] for event in record], dtype=object),
            'loss_pos_y': np.array([[
                source.position[1] for source in event.sources
            ] for event in record], dtype=object),
            'loss_pos_z': np.array([[
                source.position[2] for source in event.sources
            ] for event in record], dtype=object),
            'loss_n_photons': loss_counts
        }
        return ak.Array(d)
