# -*- coding: utf-8 -*-
# hebe.py
# Copyright (C) 2022 Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the package

# imports
import numpy as np
import h5py
import awkward as ak
import pyarrow.parquet as pq
import pyarrow 
from .utils.geo_utils import get_endcap,get_injRadius,get_volume
from .config import config
from .detector import detector_from_geo
#from .detector_handler import DH
from .photonpropagator import PP
from .lepton_prop import LP
from .lepton_injector import LepInj
# from .ppc_plotting import plot_event
from .particle import Particle
from .utils.hebe_ui import run_ui

from tqdm import tqdm
from time import time
from warnings import warn
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
        #self._dh = DH()

        self._det = detector_from_geo(config["detector"]["detector specs file"])
        print('Finished the detector')
        is_ice = config["lepton propagator"]["medium"].lower()=='ice'
        endcap = get_endcap(self._det.module_coords, is_ice)
        inj_radius = get_injRadius(self._det.module_coords, is_ice)
        cyl_radius = get_volume(self._det.module_coords, is_ice)[0]
        cyl_height = get_volume(self._det.module_coords, is_ice)[1]
        if not config["lepton injector"]["force injection params"]:
            print(
                'WARNING: Overwriting injection parameters with calculated values.'
            )
            config["lepton injector"]["simulation"]["endcap length"] = endcap
            config["lepton injector"]["simulation"]["injection radius"] = inj_radius
            config["lepton injector"]["simulation"]["cylinder radius"] = cyl_radius
            config["lepton injector"]["simulation"]["cylinder height"] = cyl_height
        if not config["lepton propagator"]["force propagation params"]:
            print('WARNING: Overwriting propagation parameters with calculated values')
            config["lepton propagator"]["propogation padding"] = inj_radius
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
        if config["lepton injector"]["inject"]:
            print('Setting up and running LI')
            if config['lepton injector']['inject']:
                print('Injecting')
                self._LI = LepInj()
            else:
                print('Not injecting')
            print('Finished LI, loading data')
        self._LI_raw = h5py.File(
            config['lepton injector']['simulation']['output name'], "r"
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
        self._new_results = {}
        self._results = {}
        self._results_record = {}
        propped_primaries = []
        for event_id, key in enumerate(self._final_states.keys()):
            print('-------------------------------')
            print('Starting set')
            self._results[key] = []
            self._results_record[key] = []
            self._new_results[key] = []
            for event in tqdm(self._final_states[key]):
                # Making sure the event id is okay:
                if event[1] in config['particles']['explicit']:
                    pdg_code = event[1]
                else:
                    pdg_code = config['particles']['replacement']
                pos = np.array(event[2])
                direction = [
                    np.cos(event[3][1])*np.sin(event[3][0]),
                    np.sin(event[3][1])*np.sin(event[3][0]),
                    np.cos(event[3][0])
                ]

                primary_particle = Particle(
                    pdg_code,
                    event[4],
                    pos,
                    direction,
                    event_id,
                    theta=event[3][0],
                    phi=event[3][1])
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
        # except ValueError:
        #     print('No hits generated, skipping dump!')
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


    def _serialize_injection_to_dict(self, LI_file, fill_dic):
        self._LI_converter = {
            tuple([12, 11,-2000001006]): 1,
            tuple([14, 13,-2000001006]): 1,
            tuple([16, 15,-2000001006]): 1,
            tuple([12, 12,-2000001006]): 2,
            tuple([14, 14,-2000001006]): 2,
            tuple([16, 16,-2000001006]): 2,
            tuple([-12, -11,-2000001006]): 1,
            tuple([-14, -13,-2000001006]): 1,
            tuple([-16, -15,-2000001006]): 1,
            tuple([-12, -12,-2000001006]): 2,
            tuple([-14, -14,-2000001006]): 2,
            tuple([-16, -16,-2000001006]): 2,
            tuple([-12, -2000001006,-2000001006]): 0,
            tuple([-12, 11,-12]): 0,
            tuple([-12, 13,-14]): 0,
            tuple([-12, 15,-16]): 0
        }
        initial_props = np.array(LI_file[config['run']['group name']]['properties'])
        interactions = np.array([[i[7], i[5], i[6]] for i in initial_props])
        initial_type = np.array([i[7] for i in initial_props])
        initial_energy = np.array([i[0] for i in initial_props])
        initial_zenith = np.array([i[1] for i in initial_props])
        initial_azimuth = np.array([i[2] for i in initial_props])
        bjorkenx = np.array([i[3] for i in initial_props])
        bjorkeny = np.array([i[3] for i in initial_props])
        injected_pos_x = np.array([i[8] for i in initial_props])
        injected_pos_y = np.array([i[9] for i in initial_props])
        injected_pos_z = np.array([i[10] for i in initial_props])
        column_depth = np.array([i[11] for i in initial_props])
        int_ids = np.array([self._LI_converter[tuple(i)] for i in interactions])
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
        fill_dic['event_id'] = events_idx
        fill_dic['mc_truth'] = {
            'injection_energy': initial_energy,
            'injection_type': initial_type,
            'injection_interaction_type': int_ids,
            'injection_zenith': initial_zenith,
            'injection_azimuth': initial_azimuth,
            'injection_bjorkenx': bjorkenx,
            'injection_bjorkeny': bjorkeny,
            'injection_position_x': injected_pos_x,
            'injection_position_y': injected_pos_y,
            'injection_position_z': injected_pos_z,
            'injection_column_depth': column_depth,
            'primary_lepton_1_type': initial_types_1,
            'primary_hadron_1_type': initial_types_2,
            'primary_lepton_1_position_x': np.array(initial_pos_1[:, 0]),
            'primary_lepton_1_position_y': np.array(initial_pos_1[:, 1]),
            'primary_lepton_1_position_z': np.array(initial_pos_1[:, 2]),
            'primary_hadron_1_position_x': np.array(initial_pos_2[:, 0]),
            'primary_hadron_1_position_y': np.array(initial_pos_2[:, 1]),
            'primary_hadron_1_position_z': np.array(initial_pos_2[:, 2]),
            'primary_lepton_1_direction_theta': np.array(initial_dir_1[:, 0]),
            'primary_lepton_1_direction_phi': np.array(initial_dir_1[:, 1]),
            'primary_hadron_1_direction_theta': np.array(initial_dir_2[:, 0]),
            'primary_hadron_1_direction_phi': np.array(initial_dir_2[:, 1]),
            'primary_lepton_1_energy': initial_energy_1,
            'primary_hadron_1_energy': initial_energy_2,
            'total_energy': initial_energy_1 + initial_energy_2
        }

    def _serialize_results_to_dict(
        self,
        results,
        record,
        fill_dic,
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
            fill_dic[particle] = {
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
                'sensor_string_id': np.array([
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

    def _construct_totals_from_dict(
            self, fill_dic
        ):
        # TODO: Remove hardcoding
        # Total
        print(type(fill_dic["primary_lepton_1"]["sensor_id"]))
        print(fill_dic["primary_lepton_1"]["sensor_id"])
        print(fill_dic["primary_hadron_1"]["sensor_id"])
        #sensor_id_all = np.concatenate(
        sensor_id_all = ak.concatenate(
            (
                fill_dic['primary_lepton_1']['sensor_id'],
                fill_dic['primary_hadron_1']['sensor_id']),
            axis=1)
        sensor_pos_all = np.array([
        self._det.module_coords[hits]
            for hits in sensor_id_all
        ], dtype=object)
        sensor_string_id_all = np.array([
            np.array(self._det._om_keys)[event]
            for event in sensor_id_all
        ], dtype=object)
        fill_dic['total'] = {
            'sensor_id': sensor_id_all,
            'sensor_pos_x': np.array([
                event[:, 0] for event in sensor_pos_all
            ], dtype=object),
            'sensor_pos_y': np.array([
                event[:, 1] for event in sensor_pos_all
            ], dtype=object),
            'sensor_pos_z': np.array([
                event[:, 2] for event in sensor_pos_all
            ], dtype=object),
            'sensor_string_id': np.array([
                event[:, 0] for event in sensor_string_id_all
            ], dtype=object),
            #'t': np.concatenate(
            't': ak.concatenate(
                (fill_dic['primary_lepton_1']['t'],
                 fill_dic['primary_hadron_1']['t']), axis=1),
        }

    def _serialize_particle_to_dict(
        self,
        particles,
        fill_dict,
        is_full,
        field_name="lepton"
    ):

        fill_dict[field_name] = {}
        tree = fill_dict[field_name]
        tree["string_id"] = [
            [hit[0] for hit in p.hits] if len(p.hits) > 0 else [-1]
            for p in particles
        ]
        tree["sensor_id"] = [
            [hit[1] for hit in p.hits] if len(p.hits) > 0 else [-1]
            for p in particles
        ]
        tree["t"] = [
            np.array([hit[2] for hit in p.hits]) if len(p.hits) > 0 else [-1]
            for p in particles
        ]
        # TODO do this for the children
        xyz = [
            np.array([self._det[(hit[0], hit[1])].pos for hit in p.hits]) if len(p.hits) > 0 else np.array([[-1, -1, -1]])
            for p in particles
        ]

        tree["sensor_pos_x"] = [
            x[:,0] for x in xyz
        ]
        tree["sensor_pos_y"] = [
            x[:,1] for x in xyz
        ]
        tree["sensor_pos_z"] = [
            x[:,2] for x in xyz
        ]

        if is_full:
            for i, particle in enumerate(particles):
                for child in particle.children:
                    tree["string_id"][i] = np.hstack(
                        (tree["string_id"][i], [hit[0] for hit in child.hits])
                    )
                    tree["sensor_id"][i] = np.hstack(
                        (tree["sensor_id"][i], [hit[1] for hit in child.hits])
                    )
                    tree["t"][i] = np.hstack(
                        (tree["t"][i], [hit[2] for hit in child.hits])
                    )
                    xyz = [
                        np.array([self._det[(hit[0], hit[1])].pos for hit in child.hits])
                    ]
                    tree["sensor_pos_x"] = np.hstack(
                        (tree["sensor_pos_x"], [x[:,0] for x in xyz])
                    )
                    tree["sensor_pos_y"] = np.hstack(
                        (tree["sensor_pos_y"], [x[:,1] for x in xyz])
                    )
                    tree["sensor_pos_z"] = np.hstack(
                        (tree["sensor_pos_z"], [x[:,2] for x in xyz])
                    )

        #return ak.Array(tree)

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
        tree = {}
        if ((sim_switch == "PPC") or (sim_switch == "PPC_CUDA")):
            self._serialize_injection_to_dict(self._LI_raw, tree)
            n = int(len(self._propped_primaries)/2)
            finals1 = self._propped_primaries[:n]
            finals2 = self._propped_primaries[n:]
            for finals in [finals1, finals2]:
                if abs(int(finals[0])) in [11, 13, 15]:
                    field_name = "primary_lepton_1"
                else:
                    field_name = "primary_hadron_1"
                self._serialize_particle_to_dict(
                    finals,
                    tree,
                    False,
                    field_name=field_name
                )
            self._construct_totals_from_dict(tree)
            tree = ak.Array(tree)
        else:
            self._serialize_injection_to_dict(self._LI_raw, tree)
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
        print("Converting to parquet")
        ak.to_parquet(
            tree,
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
        os.remove(config['lepton injector']['simulation']['output name'])
        os.remove('./config.lic')
        for key in self._results.keys():
            try:
                os.remove(
                    config['photon propagator']['storage location'] + key + '.parquet'
                )
            except FileNotFoundError:
                continue
