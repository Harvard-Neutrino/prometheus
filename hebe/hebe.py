# -*- coding: utf-8 -*-
# hebe.py
# Copyright (C) 2022 Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the package

# imports
import numpy as np
import awkward as ak

from tqdm import tqdm
from time import time
from warnings import warn
import os
import json

from jax import random  # noqa: E402

from .utils.geo_utils import get_endcap, get_injection_radius, get_volume
from .config import config
from .detector import detector_from_geo
from .photonpropagator import PP
from .lepton_prop import LP
from .injection import injection_dict
from .particle import Particle
from .utils.hebe_ui import run_ui

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

class UnknownSimulationError(Exception):
    """Raised when simulation name is not know"""
    def __init__(self, simname):
        self.message = f"Simulation name {simname} is not recognized. Only PPC and olympus"
        super().__init__(self.message)

class IncompaticleFieldsError(Exception):
    """Raised when two awkward.Array cannot be combined because fields don't match"""
    def __init__(self, fields1, fields2):
        self.message = f"Fields must fully overlap to combine. The fields were {fields1} and {fields2}"
        super().__int__(self.message)

def join_awkward_arrays(arr1, arr2, fields=None):
    # Infer fields from arrs if not passed
    if fields is None:
        if not (
            set(arr1.fields).issubset(set(arr2.fields)) and
            set(arr2.fields).issubset(set(arr1.fields))
        ):
            raise IncompaticleFieldsError(arr1.fields, arr2.fields)
        else:
            fields = arr1.fields

    arr = ak.Array(
        {
            k: [np.hstack([x, y]) for x, y in zip(getattr(arr1, k), getattr(arr2, k))]
        for k in fields}
    )

    return arr

class HEBE(object):
    """
    class: HEBE
    Interace between injection and Olympus
    Parameters
    ----------
    config : dic
        Configuration dictionary for the simulation
    Returns
    -------
    None
    """
    def __init__(self, userconfig=None):
        """
        function: __init__
        Initializes the class HEBE.
        Here all run parameters are set.
        Parameters
        ----------
        config : dic
            Configuration dictionary for the simulation
        Returns
        -------
        None
        """
        # Inputs
        start = time()
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)
        # Dumping config file
        # Has to happend before the random state is thrown in
        config["runtime"] = None
        print('-------------------------------------------')
        print('Dumping config file')
        with open(config['general']['config location'], 'w') as f:
            json.dump(config, f, indent=2)
        print('Finished dump')
        # Create RandomState
        self._is_full = config["general"]["full output"]
        if config["general"]["random state seed"] is None:
            rstate = np.random.RandomState()
            rstate_jax = random.PRNGKey(1)
        else:
            rstate = np.random.RandomState(
                config["general"]["random state seed"]
            )
            rstate_jax = random.PRNGKey(
                config["general"]["random state seed"]
            )
        config["runtime"] = {
            "random state": rstate,
            "random state jax": rstate_jax,
        }
        print('-------------------------------------------')
        # Setting up the detector
        print('-------------------------------------------')
        print('Setting up the detector')

        self._det = detector_from_geo(config["detector"]["detector specs file"])
        print('Finished the detector')
        print('-------------------------------------------')
        print('Setting up lepton propagation')
        self._lp = LP()
        print('Finished the lepton propagator')
        # Photon propagation
        print('-------------------------------------------')
        # Setting up the photon propagation
        print('-------------------------------------------')
        print('Setting up photon propagation')
        self._pp = PP(self._lp, self._det)
        print('Finished the photon propagator')
        # Photon propagation
        print('-------------------------------------------')
        end = time()
        print(
            'Setup and preliminary ' +
            'simulations took %f seconds' % (end - start))
        print('-------------------------------------------')

    @property
    def results(self):
        """ Returns the results from the simulation
        """
        return self._results

    @property
    def results_record(self):
        """ Returns the record results from the simulation
        """
        return self._results_record

    def injection(self):
        """ Injects leptons according to the config file
        """
        import h5py
        # Loading injection data
        print('-------------------------------------------')
        start = time()
        injection_config = config["injection"][config["injection"]["name"]]
        self._injection = injection_dict[config["injection"]["name"]](
            injection_config['paths']['output name']
        )
        print('Setting up and running injection')
        if injection_config['inject']:
            injection_config["simulation"]["random state seed"] = config["general"]["random state seed"]
            if not injection_config["simulation"]["force injection params"]:
                warn(
                    'WARNING: Overwriting injection parameters with calculated values.'
                )
                is_ice = config["lepton propagator"]["medium"].lower() == 'ice'
                endcap = get_endcap(self._det.module_coords, is_ice)
                inj_radius = get_injection_radius(self._det.module_coords, is_ice)
                cyl_radius, cyl_height = get_volume(self._det.module_coords, is_ice)
                injection_config["simulation"]["endcap length"] = endcap
                injection_config["simulation"]["injection radius"] = inj_radius
                injection_config["simulation"]["cylinder radius"] = cyl_radius
                injection_config["simulation"]["cylinder height"] = cyl_height
            print('Injecting')
            self._injection.inject(injection_config)
        else:
            print('Not injecting')
        self._injection.load_data()
        print('Finished injection, loading data')
        print('Finished loading')
        # Creating data set
        print('Creating the data set for further propagation')
        print('Finished the data set')
        end = time()
        print(
            'Injection ' +
            'simulations took %f seconds' % (end - start))
        print('-------------------------------------------')
        print('-------------------------------------------')

    def propagate(self):
        """ Runs the light yield calculations
        """
        print('-------------------------------------------')
        # Simulation loop
        print('-------------------------------------------')
        print('Starting particle loop')
        start = time()
        self._new_results = {}
        self._results = {}
        self._results_record = {}
        propped_primaries = []
        if config["run"]["subset"]["switch"]:
            nevents = config["run"]["subset"]["counts"]
        else:
            nevents = len(self._injection)
        #for event_id, key in enumerate(self._final_states.keys()):
        for key in ["primary_lepton_1", "primary_hadron_1"]:
            print('-------------------------------')
            print('Starting set')
            self._results[key] = []
            self._results_record[key] = []
            self._new_results[key] = []
            # TODO load the injection into an event structure
            for event_idx in tqdm(range(nevents)):
            #for event in tqdm(self._final_states[key]):
                # Making sure the event id is okay:
                pdg_code = getattr(self._injection, f"{key}_type")[event_idx]
                # TODO what is the point of this ??????
                if not pdg_code in config['particles']['explicit']:
                    pdg_code = config['particles']['replacement']
                pos = np.array([
                    getattr(self._injection, f"{key}_position_x")[event_idx],
                    getattr(self._injection, f"{key}_position_y")[event_idx],
                    getattr(self._injection, f"{key}_position_x")[event_idx],
                ])
                zen = getattr(self._injection, f"{key}_direction_theta")[event_idx]
                azi = getattr(self._injection, f"{key}_direction_phi")[event_idx]
                direction = [
                    np.cos(azi)*np.sin(zen),
                    np.sin(azi)*np.sin(zen),
                    np.cos(zen)
                ]

                primary_particle = Particle(
                    pdg_code,
                    getattr(self._injection, f"{key}_energy")[event_idx],
                    pos,
                    direction,
                    #event_id,
                    # TODO why the fuck is there two reps of direction ?
                    theta=zen,
                    phi=azi
                )
                res_event, res_record = self._pp._sim(primary_particle)
                propped_primaries.append(primary_particle)
                self._new_results[key].append(primary_particle)
                self._results[key].append(res_event)
                self._results_record[key].append(res_record)
            print('-------------------------------')
        self._propped_primaries = propped_primaries
        print('Finished particle loop')
        print('-------------------------------------------')
        end = time()
        print(
            'The simulation ' +
            'took %f seconds' % (end - start))
        print('-------------------------------------------')
        print('-------------------------------------------')
        print('Results are stored in self.results')
        print('-------------------------------------------')

    def sim(self):
        """ Utility function to run all steps of the simulation
        """
        self.injection()
        self.propagate()
        print('Dumping results')
        self.construct_output(
            sim_switch=config['photon propagator']['name']
        )
        config["runtime"] = None
        print('Finished dump')
        if config["general"]["clean up"]:
            print('-------------------------------------------')
            print('-------------------------------------------')
            self._clean_up()
            print("Finished cleaning")

        print('Have a good day!')
        print('-------------------------------------------')
        print('-------------------------------------------')

    def _serialize_results_to_dict(
        self,
        results,
        record,
        fill_dict,
        particle="lepton"
    ):
        # TODO: Optimize this. Currently this is extremely inefficient.
            # first set
            all_ids_1 = []
            for event in results:
                dom_ids_1 = []
                for id_dom, dom in enumerate(event):
                    if len(dom) > 0:
                        dom_ids_1.append([id_dom] * len(dom))
                dom_ids_1 = ak.flatten(ak.Array(dom_ids_1), axis=None)
                all_ids_1.append(dom_ids_1)
            all_ids_1 = ak.Array(all_ids_1)
            all_hits_1 =  []
            for event in results:
                all_hits_1.append(ak.flatten(event, axis=None))
            all_hits_1 = ak.Array(all_hits_1)
            # Positional sensor information
            sensor_pos_1 = np.array([
                self._det.module_coords[hits]
                for hits in all_ids_1
            ], dtype=object)
            sensor_string_id_1 = np.array([
                np.array(self._det._om_keys)[event]
                for event in all_ids_1
            ], dtype=object)
            # The losses
            loss_counts = np.array([[
                source.n_photons[0] for source in event.sources
            ] for event in record], dtype=object)
            # This is as inefficient as possible
            fill_dict[particle] = {
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

    def _totals_from_awkward_arr(
            self,
            arr
        ):

        # These are the keys which refer to the physical particles
        particle_fields = [
            field for field in arr.fields
            if field not in "event_id mc_truth".split()
        ]

        # Return `None` if no particles made light
        if len(particle_fields)==0:
            return None

        outarr = getattr(arr, particle_fields[0])
        for field in particle_fields[1:]:
            outarr = join_awkward_arrays(outarr, getattr(arr, field))
        return outarr

    def _construct_totals_from_dict(
            self,
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
                self._det.module_coords[hits]
                for hits in sensor_id_all
            ],
            dtype=object
        )
        sensor_string_id_all = np.array(
            [
                np.array(self._det._om_keys)[event]
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

    def construct_output(
            self,
            sim_switch="olympus"
        ):
        """ Constructs a parquet file with metadata from the generated files.
        Currently this still treats olympus and ppc output differently.
        Parameters
        ----------
        sim_switch: str
            switch for olympus or ppc mode
        Notes
        -----
        It contains the fields:
            ['event_id',
             'mc_truth'
             'lepton',
             'hadron',
             'total]
        Load the file using the awkward method from_parquet
        mc_truth, lepton, hadron and total in turn contain subfields.
        """
        # TODO: Unify this for olympus and PPC
        print("Generating output for a " + sim_switch + " simulation.")
        print("Generating the different particle fields...")
        sdsd = 0
        outarr = ak.Array({})
        sdsd = 1
        if "ppc" in sim_switch.lower():
            outarr = ak.with_field(
                outarr,
                self._injection.serialize_to_awkward(),
                where="mc_truth"
            )
            lepton_idx = 1
            hadron_idx = 1
            n = int(len(self._propped_primaries) / 2)
            finals_1 = self._propped_primaries[:n]
            finals_2 = self._propped_primaries[n:]
            for finals in [finals_1, finals_2]:
                if abs(int(finals[0])) in [11, 13, 15]:
                    field_name = f"primary_lepton_{lepton_idx}"
                    lepton_idx += 1
                else:
                    field_name = f"primary_hadron_{hadron_idx}"
                    hadron_idx += 1
                test_arr = self._serialize_particle_to_awkward(finals)
                # We only add this to the array if anything made light
                if test_arr is not None:
                    outarr = ak.with_field(
                        outarr,
                        test_arr,
                        where=field_name
                    )
            test_arr = self._totals_from_awkward_arr(outarr)
            if test_arr is not None:
                outarr = ak.with_field(
                    outarr,
                    test_arr,
                    where="total"
                )
            tree = outarr
        elif sim_switch=="olympus":
            self._serialize_injection_to_dict(self._injection_raw, tree)
            # Looping over primaries, change hardcoding
            starting_particles = ['primary_lepton_1', 'primary_hadron_1']
            try:
                for i, primary in enumerate(starting_particles):
                    self._serialize_results_to_dict(
                        self._results['final_%d' % (i + 1)],
                        self._results_record['final_%d' % (i + 1)],
                        tree,
                        primary
                    )
                self._construct_totals_from_dict(tree)
            except:
                warn("No hits generated!")
            tree = ak.Array(tree)
        else:
            raise UnknownSimulationError(sim_switch)
        print("Converting to parquet")
        ak.to_parquet(
            tree,
            config['photon propagator']['storage location'] +
            config['general']['meta name'] + '.parquet'
        )
        print("Adding metadata")
        # Adding meta data
        import pyarrow.parquet as pq
        table = pq.read_table(
            config['photon propagator']['storage location'] +
            config['general']['meta name'] + '.parquet')
        config['runtime'] = None
        custom_meta_data = json.dumps(config)
        custom_meta_data_key = 'config_prometheus'
        combined_meta = {
            custom_meta_data_key.encode() : custom_meta_data.encode()
        }
        table = table.replace_schema_metadata(combined_meta)
        pq.write_table(
            table,
            config['photon propagator']['storage location'] +
            config['general']['meta name'] + '.parquet'
        )
        print("Finished new data file")

    def plot(self, event, **kwargs):
        plot_event(event, self._det, **kwargs)

    def event_plotting(
            self, det, event, record=None,
            plot_tfirst=False, plot_hull=False):
        """ utilizes olympus plotting to generate a plot
        Parameters
        ----------
        det : Detector
            A detector object
        event : awkward.array
            The resulting events from the simulation
        record : awkward.array
            Record of the results from the simulation
        """
        self._pp._plotting(
            det, event, record=record,
            plot_tfirst=plot_tfirst, plot_hull=plot_hull)

    def __del__(self):
        """ What to do when the hebe instance is deleted
        """
        print("I am melting.... AHHHHHH!!!!")

    def _clean_up(self):
        """ Remove temporary and intermediate files.
        """
        print("Removing intermediate data files.")
        os.remove(config['injection']['simulation']['output name'])
        os.remove('./config.lic')
        for key in self._results.keys():
            try:
                os.remove(
                    config['photon propagator']['storage location'] + key + '.parquet'
                )
            except FileNotFoundError:
                continue
