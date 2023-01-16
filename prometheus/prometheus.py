# -*- coding: utf-8 -*-
# prometheus.py
# Copyright (C) 2022 Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the package

import numpy as np
import awkward as ak
import pyarrow.parquet as pq
import os
import json
from typing import Union
from tqdm import tqdm
from time import time
from jax import random  # noqa: E402

from .utils import config_mims, clean_config, totals_from_awkward_arr
from .config import config
from .detector import Detector
from .photon_propagation import photon_propagator
from .injection import INJECTION_DICT, particle_from_injection

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

class UnknownSimulationError(Exception):
    """Raised when simulation name is not know"""
    def __init__(self, simname):
        self.message = f"Simulation name {simname} is not recognized. Only PPC and olympus"
        super().__init__(self.message)

class CannotLoadDetectorError(Exception):
    """Raised when detector not provided and cannot be determined from config"""
    def __init__(self):
        self.message = f"No Detector provided and no geo file path given in config"
        super().__init__(self.message)

class Prometheus(object):
    """Class for unifying injection, energy loss calculation, and photon propagation"""
    def __init__(
        self,
        userconfig: Union[None, dict, str]=None,
        detector: Union[None, Detector]=None
    ) -> None:
        """Initializes the Prometheus class

        params
        ______
        userconfig: Configuration dictionary or path to yaml file which specifies configuration
        detector: Detector to be used or path to geo file to load detector file. If this is left
            out, the path from the `userconfig["detector"]["specs file"]` be loaded

        raises
        ______
        CannotLoadDetectorError: When no detector provided and no geo file path provided in config

        """
        start = time()
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        # Set up the detector
        if detector is None and config["detector"]["specs file"] is None:
            raise CannotLoadDetectorError()

        if detector is None:
            print(f"Building detector from {config['detector']['specs file']}")
            from .detector import detector_from_geo
            detector = detector_from_geo(config["detector"]["specs file"])
        
        self._detector = detector

        # Infer which config to use from the PROPOSAL version
        # We need to check the version prior to import, otherwise
        # the type hinting will throw an error
        import proposal as pp
        if int(pp.__version__.split(".")[0]) <= 6:
            from .lepton_propagation import OldProposalLeptonPropagator as LP
            config["lepton propagator"]["name"] = "old proposal"
        else:
            from .lepton_propagation import NewProposalLeptonPropagator as LP
            config["lepton propagator"]["name"] = "new proposal"
        config["lepton propagator"]["version"] = pp.__version__

        # Make config internally consistent
        config_mims(config, self.detector)
        # Remove unused fields from config
        clean_config(config)
        # Configure the injection
        injection_config = config["injection"][config["injection"]["name"]]
        self._injection = INJECTION_DICT[config["injection"]["name"]](
            injection_config["paths"]["output name"]
        )
        # Configure the lepton propagator
        lp_config = config["lepton propagator"][config["lepton propagator"]["name"]]
        self._lepton_propagator = LP(lp_config)
        # Configure the photon propagator
        pp_config = config["photon propagator"][config["photon propagator"]["name"]]
        photon_propagator_class = photon_propagator(config["photon propagator"]["name"])
        self._pp = photon_propagator_class(self._lepton_propagator, self.detector, pp_config)
        end = time()
        print(f"Setup and preliminary simulations took {end - start} seconds")
        print("-------------------------------------------")

    @property
    def detector(self):
        return self._detector

    @property
    def injection(self):
        return self._injection

    @property
    def results(self):
        """Returns the results from the simulation"""
        return self._results

    @property
    def results_record(self):
        """Returns the record results from the simulation"""
        return self._results_record

    def inject(self):
        """Determines initial neutrino and final particle states according to config"""
        print('-------------------------------------------')
        start = time()
        injection_config = config["injection"][config["injection"]["name"]]
        if injection_config["inject"]:
            injection_config["simulation"]["random state seed"] = config["general"]["random state seed"]
            self._injection.inject(
                injection_config,
                detector_offset=self.detector.offset
            )
        else:
            print("Not injecting")
        self._injection.load_data()
        print("Finished injection, loading data")
        print("Finished loading")
        # Creating data set
        print("Creating the data set for further propagation")
        print("Finished the data set")
        end = time()
        print(f"Injection simulations took {end-start} seconds")
        print("-------------------------------------------")
        print("-------------------------------------------")

    # TODO this is psycho.
    # We should factor out generating losses and photon prop
    def propagate(self):
        """Calculates energy losses, generates photon yields, and propagates photons"""
        print("-------------------------------------------")
        print("-------------------------------------------")
        print("Starting particle loop")
        start = time()
        self._new_results = {}
        self._results = {}
        self._results_record = {}
        propped_primaries = []
        if config["run"]["subset"]["switch"]:
            nevents = config["run"]["subset"]["counts"]
        else:
            nevents = len(self._injection)
        for key in ["primary_particle_1", "primary_particle_2"]:
            print("-------------------------------")
            print("Starting set")
            self._results[key] = []
            self._results_record[key] = []
            self._new_results[key] = []
            for event_idx in tqdm(range(nevents)):
                primary_particle = particle_from_injection(
                    self._injection,
                    key,
                    event_idx
                )
                res_event, res_record = self._pp.propagate(primary_particle)
                propped_primaries.append(primary_particle)
                self._new_results[key].append(primary_particle)
                self._results[key].append(res_event)
                self._results_record[key].append(res_record)
            print("-------------------------------")
        self._propped_primaries = propped_primaries
        end = time()
        print("Finished particle loop")
        print("-------------------------------------------")
        print(f"The simulation took {end - start} seconds")
        print("-------------------------------------------")
        print("-------------------------------------------")
        print("Results are stored in self.results")
        print("-------------------------------------------")

    def sim(self):
        """Performs injection of precipitating interaction, calculates energy losses,
        calculates photon yield, propagates photons, and save resultign photons"""
        # Has to happen before the random state is thrown in
        if "runtime" in config["photon propagator"].keys():
            config["photon propagator"]["runtime"] = None
        print("-------------------------------------------")
        print("Dumping config file")
        with open(config["general"]["config location"], "w") as f:
            json.dump(config, f, indent=2)
        print("Finished dump")
        # add random state stuff to this olympus dict. Seems bizarre but w.e.
        if config["photon propagator"]["name"].lower()=="olympus":
            rstate = np.random.RandomState(config["general"]["random state seed"])
            rstate_jax = random.PRNGKey(config["general"]["random state seed"])
            # TODO this feels like it shouldn't be in the config
            config["photon propagator"]["olympus"]["runtime"] = {
                "random state": rstate,
                "random state jax": rstate_jax,
            }
        self.inject()
        self.propagate()
        print("Dumping results")
        self.construct_output()
        config["runtime"] = None
        print("Finished dump")
        if config["general"]["clean up"]:
            print("-------------------------------------------")
            print("-------------------------------------------")
            self._clean_up()
            print("Finished cleaning")

        print("Have a good day!")
        print("-------------------------------------------")
        print("-------------------------------------------")

    def construct_output(self):
        """Constructs a parquet file with metadata from the generated files.
        Currently this still treats olympus and ppc output differently."""
        sim_switch = config["photon propagator"]["name"]
        if not ("ppc" in sim_switch.lower() or sim_switch.lower()=="olympus"):
            raise UnknownSimulationError(sim_switch)

        # TODO: Unify this for olympus and PPC
        print(f"Generating output for a {sim_switch} simulation.")
        print("Generating the different particle fields...")
        outarr = ak.Array({})
        # Save the injection
        outarr = ak.with_field(
            outarr,
            self._injection.serialize_to_awkward(),
            where="mc_truth"
        )
        if "ppc" in sim_switch.lower():
            from .utils import serialize_particles_to_awkward
            n = int(len(self._propped_primaries) / 2)
            finals_1 = self._propped_primaries[:n]
            finals_2 = self._propped_primaries[n:]
            for idx, finals in enumerate([finals_1, finals_2]):
                field_name = f"primary_particle_{idx+1}"
                test_arr = serialize_particles_to_awkward(self.detector, finals)
                # We only add this to the array if anything made light
                if test_arr is not None:
                    outarr = ak.with_field(outarr, test_arr, where=field_name)
        elif sim_switch.lower() == "olympus":
            from .utils import serialize_results_to_dict
            # Looping over primaries, change hardcoding
            starting_particles = ["primary_particle_1", "primary_particle_2"]
            for primary in starting_particles:
                test_arr = serialize_results_to_dict(
                    self.detector,
                    self._results[primary],
                    self._results_record[primary],
                )
                if test_arr is not None:
                    outarr = ak.with_field(outarr, test_arr, where=primary)
        test_arr = totals_from_awkward_arr(outarr)
        if test_arr is not None:
            outarr = ak.with_field(outarr, test_arr, where="total")
        print("Converting to parquet")
        ak.to_parquet(
            outarr,
            f"{config['general']['storage location']}{config['general']['meta name']}.parquet"
        )
        print("Adding metadata")
        # Adding meta data
        table = pq.read_table(
            config["general"]["storage location"] +
            config["general"]["meta name"] + ".parquet"
        )
        # TODO this is a bad way of doing this
        if config["photon propagator"]["name"].lower()=="olympus":
            if "runtime" in config["photon propagator"]["olympus"].keys():
                config["photon propagator"]["olympus"]["runtime"] = None
        custom_meta_data = json.dumps(config)
        custom_meta_data_key = "config_prometheus"
        combined_meta = {
            custom_meta_data_key.encode() : custom_meta_data.encode()
        }
        table = table.replace_schema_metadata(combined_meta)
        pq.write_table(
            table,
            f"{config['general']['storage location']}{config['general']['meta name']}.parquet"
        )
        print("Finished new data file")

    def plot(self, event, **kwargs):
        plot_event(event, self._det, **kwargs)

    def event_plotting(
        self, det, event, record=None,
        plot_tfirst=False, plot_hull=False
    ):
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
        """What to do when the Prometheus instance is deleted
        """
        print("I am melting.... AHHHHHH!!!!")

    def _clean_up(self):
        """Remove temporary and intermediate files.
        """
        print("Removing intermediate data files.")
        injector_name = config["injection"]["name"]
        os.remove(config["injection"][injector_name]["paths"]["output name"])
        for key in self._results.keys():
            try:
                os.remove(
                    f"{config['general']['storage location']}{key}.parquet"
                )
            except FileNotFoundError:
                continue
