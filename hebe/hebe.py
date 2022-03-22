# -*- coding: utf-8 -*-
# hebe.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the package

# imports
import numpy as np
import h5py
import awkward as ak
import pyarrow.parquet as pq
from .config import config
from .detector_handler import DH
from .photonpropagator import PP
from .lepton_prop import LP
from .lepton_injector import LepInj
from .ppc_plotting import plot_event

from tqdm import tqdm
from time import time
import os
import json
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from jax import random  # noqa: E402


class HEBE(object):
    """
    class: HEBE
    Interace between LI and Olympus
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
        self._dh = DH()
        self._det = self._dh.from_f2k()
        print('Finished the detector')
        # Setting up the lepton propagator
        print('-------------------------------------------')
        print('Setting up leptopn propagation')
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
        # Loading LI data
        print('-------------------------------------------')
        start = time()
        print('Setting up and running LI')
        if config['lepton injector']['inject']:
            print('Injecting')
            self._LI = LepInj()
        else:
            print('Not injection')
        print('Finished LI, loading data')
        self._LI_raw = h5py.File(
            config['lepton injector']['simulation']['output name']
        )
        self._final_states = {}
        for elem in config['run']['data sets']:
            if elem[0]:
                self._final_states[elem[1]] = (
                    self._LI_raw[config['run']['group name']][elem[1]]
                )
        print('Finished loading')
        # Creating data set
        print('Creating the data set for further propagation')
        if config['run']['subset']['switch']:
            print('Creating subset')
            for key in self._final_states.keys():
                self._final_states[key] = self._final_states[key][
                    0:config['run']['subset']['counts']
                ]
        else:
            print('Using the full data set')
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
        self._results = {}
        self._results_record = {}
        for key in self._final_states.keys():
            print('-------------------------------')
            print('Starting set')
            self._results[key] = []
            self._results_record[key] = []
            for event in tqdm(self._final_states[key]):
                # Making sure the event id is okay:
                if event[1] in config['particles']['explicit']:
                    event_id = event[1]
                else:
                    event_id = config['particles']['replacement']
                pos = np.array(event[2])
                # Introducing an injection offset caused by the different
                # coordinate systems between LI and the t2k file.
                pos = pos + np.array(config['detector']['injection offset'])
                injection_event = {
                    "time": 0.,
                    "theta": event[3][0],
                    "phi": event[3][1],
                    # TODO: This needs to be removed once the coordinate
                    # systems match!
                    "pos": pos,
                    "energy": event[4],
                    "particle_id": event_id,
                    'length': config['lepton propagator']['track length'],
                    'event id': event_id
                }
                res_event, res_record = self._pp._sim(injection_event)
                self._results[key].append(res_event)
                self._results_record[key].append(res_record)
            print('-------------------------------')
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
        if config['photon propagator']['name'] == 'olympus':
            for key in self._results.keys():
                try:
                    ak.to_parquet(
                        self._results[key],
                        config['photon propagator']['storage location'] + key + '.parquet'
                    )
                except ValueError:
                    print('No hits generated, skipping dump!')
            if config['general']['meta data file']:
                print('Storing metadata file')
                self.construct_meta_data_set(
                    config['lepton injector']['simulation']['output name'],
                    [
                        config['photon propagator']['storage location'] + key + '.parquet'
                        for key in self._results.keys()
                    ]
                )
                print('Finished meta data file')
        else:
            if config['general']['meta data file']:
                print('Storing meta data')
                self.construct_meta_data_set_ppc(
                    config['lepton injector']['simulation']['output name']
                )
        config["runtime"] = None
        print('Finished dump')
        print('Have a good day!')
        print('-------------------------------------------')
        print('-------------------------------------------')

    def construct_meta_data_set_ppc(self, LI_file: str):
        """ Constructs a parquet file with metadata from the generated files.
        Unlike the olympus version this uses the internal results.

        Parameters
        ----------
        LI_file: str
            Location of the LI file

        Notes
        -----
        It contains the fields: 
            ['event_id',
            'initial_type',
            'initial_position',
            'initial_direction',
            'initial_energy',
            'photons']
        Load the file using the awkward method from_parquet
        You can access them via the fields method from awkward arrays.
        'photons' then contains 4 subsets: dom_ids_1, dom_ids_2, t_1, t_2.
        The first two are the hit doms for the event, while the second two
        record the time.
        """
        LI_file = h5py.File(LI_file)
        # TODO: Currently the names are hardcoded. This should be changed
        initial_types_1 = np.array([i[1] for i in LI_file[config['run']['group name']]['final_1'][:]])
        initial_types_2 = np.array([i[1] for i in LI_file[config['run']['group name']]['final_2'][:]])
        initial_pos_1 = np.array([i[2] for i in LI_file[config['run']['group name']]['final_1'][:]])
        initial_pos_2 = np.array([i[2] for i in LI_file[config['run']['group name']]['final_2'][:]])
        initial_dir_1 = np.array([i[3] for i in LI_file[config['run']['group name']]['final_1'][:]])
        initial_dir_2 = np.array([i[3] for i in LI_file[config['run']['group name']]['final_2'][:]])
        initial_energy_1 = np.array([i[4] for i in LI_file[config['run']['group name']]['final_1'][:]])
        initial_energy_2 = np.array([i[4] for i in LI_file[config['run']['group name']]['final_2'][:]])
        events_idx = np.array(range(len(initial_types_1)))
        # TODO: Optimize this. Currently this is extremely inefficient.
        # first set
        all_ids_dic = {}
        all_times_dic = {}
        all_wavelength_dic = {}
        all_dom_hit_points_dic = {}
        all_photon_dir_dic = {}
        for key in self._results.keys():
            all_ids = []
            all_times = []
            all_wavelength = []
            all_dom_hit_points = []
            all_photon_dir = []
            for event in self._results[key]:
                dom_ids = []
                times = []
                wavelengths = []
                dom_hit_point = []  # Zenith azimuth in radians
                photon_dir = []  # Zenith azimuth in radians
                if len(event) > 0:
                    for hit in event:
                        dom_ids.append(hit[1])
                        times.append(hit[2])
                        wavelengths.append(hit[3])
                        dom_hit_point.append([hit[4], hit[5]])
                        photon_dir.append([[hit[6], hit[7]]])
                # No hits
                else:
                    dom_ids.append([])
                    times.append([])
                    wavelengths.append([])
                    dom_hit_point.append([])
                    photon_dir.append([])
                all_ids.append(dom_ids)
                all_times.append(times)
                all_wavelength.append(wavelengths)
                all_dom_hit_points.append(dom_hit_point)
                all_photon_dir.append(photon_dir)
            all_ids_dic[key] = all_ids
            all_times_dic[key] = all_times
            all_wavelength_dic[key] = all_wavelength
            all_dom_hit_points_dic[key] = all_dom_hit_points
            all_photon_dir_dic[key] = all_photon_dir
        # Combining
        comb_type = np.stack((initial_types_1, initial_types_2), axis = 1)
        comb_pos = np.stack((initial_pos_1, initial_pos_2), axis = 1)
        comb_dir = np.stack((initial_dir_1, initial_dir_2), axis = 1)
        comb_energy = np.stack((initial_energy_1, initial_energy_2), axis = 1)
        # TODO: Remove hard coding
        meta_a = ak.Array({
            'event_id': events_idx,
            'mc_truth': {
                'type': comb_type,
                'position': comb_pos,
                'direction': comb_dir,
                'energy': comb_energy,
            },
            'photons_1': {
                'sensor_id': all_ids_dic['final_1'],
                't': all_times_dic['final_1'],
                'wave': all_wavelength_dic['final_1'],
                'sensor_hit_point': all_dom_hit_points_dic['final_1'],
                'photon_dir': all_photon_dir_dic['final_1'],
            },
            'photons_2': {
                'sensor_id': all_ids_dic['final_2'],
                't': all_times_dic['final_2'],
                'wave': all_wavelength_dic['final_2'],
                'sensor_hit_point': all_dom_hit_points_dic['final_2'],
                'photon_dir': all_photon_dir_dic['final_2']
            }
        })
        ak.to_parquet(
            meta_a,
            config['photon propagator']['storage location'] +
            config['general']['meta name'] + '.parquet'
        )
        print("Adding metadata")
        # Adding meta data
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

    def construct_meta_data_set(self, LI_file: str, PP_files: str):
        """ Constructs a parquet file with metadata from the generated files.

        Parameters
        ----------
        LI_file: str
            Location of the LI file
        PP_files: list
            List containing the PP output locations

        Notes
        -----
        It contains the fields: 
            ['event_id',
            'initial_type',
            'initial_position',
            'initial_direction',
            'initial_energy',
            'photons']
        Load the file using the awkward method from_parquet
        You can access them via the fields method from awkward arrays.
        'photons' then contains 4 subsets: dom_ids_1, dom_ids_2, t_1, t_2.
        The first two are the hit doms for the event, while the second two
        record the time.
        """
        LI_file = h5py.File(LI_file)
        results1 = ak.from_parquet(
            PP_files[0]
        )
        results2 = ak.from_parquet(
            PP_files[1]
        )
        # TODO: Currently the names are hardcoded. This should be changed
        initial_types_1 = np.array([i[1] for i in LI_file[config['run']['group name']]['final_1'][:]])
        initial_types_2 = np.array([i[1] for i in LI_file[config['run']['group name']]['final_2'][:]])
        initial_pos_1 = np.array([i[2] for i in LI_file[config['run']['group name']]['final_1'][:]])
        initial_pos_2 = np.array([i[2] for i in LI_file[config['run']['group name']]['final_2'][:]])
        initial_dir_1 = np.array([i[3] for i in LI_file[config['run']['group name']]['final_1'][:]])
        initial_dir_2 = np.array([i[3] for i in LI_file[config['run']['group name']]['final_2'][:]])
        initial_energy_1 = np.array([i[4] for i in LI_file[config['run']['group name']]['final_1'][:]])
        initial_energy_2 = np.array([i[4] for i in LI_file[config['run']['group name']]['final_2'][:]])
        events_idx = np.array(range(len(initial_types_1)))
        # TODO: Optimize this. Currently this is extremely inefficient.
        # first set
        all_ids_1 = []
        for event in results1:
            dom_ids_1 = []
            for id_dom, dom in enumerate(event):
                if len(dom) > 0:
                    dom_ids_1.append([id_dom] * len(dom))
            dom_ids_1 = ak.flatten(ak.Array(dom_ids_1), axis=None)
            all_ids_1.append(dom_ids_1)
        all_ids_1 = ak.Array(all_ids_1)
        all_hits_1 =  []
        for event in results1:
            all_hits_1.append(ak.flatten(event, axis=None))
        all_hits_1 = ak.Array(all_hits_1)
        # Second set
        all_ids_2 = []
        for event in results2:
            dom_ids_2 = []
            for id_dom, dom in enumerate(event):
                if len(dom) > 0:
                    dom_ids_2.append([id_dom] * len(dom))
            dom_ids_2 = ak.flatten(ak.Array(dom_ids_2), axis=None)
            all_ids_2.append(dom_ids_2)
        all_ids_2 = ak.Array(all_ids_2)
        all_hits_2 =  []
        for event in results1:
            all_hits_2.append(ak.flatten(event, axis=None))
        all_hits_2 = ak.Array(all_hits_2)
        # Combining
        comb_type = np.stack((initial_types_1, initial_types_2), axis = 1)
        comb_pos = np.stack((initial_pos_1, initial_pos_2), axis = 1)
        comb_dir = np.stack((initial_dir_1, initial_dir_2), axis = 1)
        comb_energy = np.stack((initial_energy_1, initial_energy_2), axis = 1)

        meta_a = ak.Array({
            'event_id': events_idx,
            'mc_truth': {
                'type': comb_type,
                'position': comb_pos,
                'direction': comb_dir,
                'energy': comb_energy,
            },
            'photons_1': {
                'sensor_id': all_ids_1,
                't': all_hits_1,
            },
            'photons_2': {
                'sensor_id': all_ids_2,
                't': all_hits_2,
            }
        })
        ak.to_parquet(
            meta_a,
            config['photon propagator']['storage location'] +
            config['general']['meta name'] + '.parquet'
        )
        print("Adding metadata")
        # Adding meta data
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

    def ppc_event_plotting(self, event, **kwargs):
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
