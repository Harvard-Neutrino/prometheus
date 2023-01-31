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

from .utils import config_mims, clean_config
from .config import config
from .detector import Detector
from .injection import RegisteredInjectors, INJECTION_CONSTRUCTOR_DICT
from .lepton_propagation import RegisteredLeptonPropagators
from .photon_propagation import (
    get_photon_propagator,
    RegisteredPhotonPropagators
)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

class CannotLoadDetectorError(Exception):
    """Raised when detector not provided and cannot be determined from config"""
    def __init__(self):
        self.message = f"No Detector provided and no geo file path given in config"
        super().__init__(self.message)

def regularize(s: str) -> str:
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.upper()
    return s

class Prometheus(object):
    """Class for unifying injection, energy loss calculation, and photon propagation"""
    def __init__(
        self,
        userconfig: Union[None, dict, str] = None,
        detector: Union[None, Detector] = None
    ) -> None:
        """Initializes the Prometheus class

        params
        ______
        userconfig: Configuration dictionary or path to yaml file 
            which specifies configuration
        detector: Detector to be used or path to geo file to load detector file.
            If this is left out, the path from the `userconfig["detector"]["specs file"]`
            be loaded

        raises
        ______
        CannotLoadDetectorError: When no detector provided and no
            geo file path provided in config

        """
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        if regularize(config["injection"]["name"]) not in RegisteredInjectors.list():
            # TODO make this error
            raise UnknownInjectorError(config["injection"]["name"])

        if regularize(config["lepton propagator"]["name"]) not in RegisteredLeptonPropagators.list():
            # TODO make this error
            raise UnknownLeptonPropagatorError()

        if regularize(config["photon propagator"]["name"]) not in RegisteredPhotonPropagators.list():
            # TODO make this error
            raise UnknownPhotonPropagatorError()

        if detector is None and config["detector"]["specs file"] is None:
            raise CannotLoadDetectorError()

        if detector is None:
            from .detector import detector_from_geo
            detector = detector_from_geo(config["detector"]["specs file"])

        self._injector = getattr(
            RegisteredInjectors,
            regularize(config["injection"]["name"])
        )
        self._lp = getattr(
            RegisteredLeptonPropagators,
            regularize(config["lepton propagator"]["name"])
        )
        self._pp = getattr(
            RegisteredPhotonPropagators,
            regularize(config["photon propagator"]["name"])
        )
        
        self._detector = detector
        self._injection = None

        # Infer which config to use from the PROPOSAL version
        # We need to check the version prior to import, otherwise
        # the type hinting will throw an error
        import proposal as pp
        if int(pp.__version__.split(".")[0]) <= 6:
            from .lepton_propagation import OldProposalLeptonPropagator as LeptonPropagator
            config["lepton propagator"]["name"] = "old proposal"
        else:
            from .lepton_propagation import NewProposalLeptonPropagator as LeptonPropagator
            config["lepton propagator"]["name"] = "new proposal"
        config["lepton propagator"]["version"] = pp.__version__

        config_mims(config, self.detector)
        clean_config(config)

        lepton_prop_config = config["lepton propagator"][config["lepton propagator"]["name"]]
        self._lepton_propagator = LeptonPropagator(lepton_prop_config)

        pp_config = config["photon propagator"][config["photon propagator"]["name"]]
        self._photon_propagator = get_photon_propagator(config["photon propagator"]["name"])(
            self._lepton_propagator,
            self.detector,
            pp_config
        )

    @property
    def detector(self):
        return self._detector

    @property
    def injection(self):
        if self._injection is None:
            # TODO Make this error
            raise NoInjectionError()
        return self._injection

    def inject(self):
        """Determines initial neutrino and final particle states according to config"""
        injection_config = config["injection"][config["injection"]["name"]]
        if injection_config["inject"]:

            from .injection import INJECTOR_DICT
            if self._injector not in INJECTOR_DICT.keys():
                # TODO make this error
                raise InjectorNotImplementedError()

            injection_config["simulation"]["random state seed"] = (
                config["run"]["random state seed"]
            )

            INJECTOR_DICT[self._injector](
                injection_config["paths"],
                injection_config["simulation"],
                detector_offset=self.detector.offset
            )

        self._injection = INJECTION_CONSTRUCTOR_DICT[self._injector](
            injection_config["paths"]["injection file"]
        )

    # We should factor out generating losses and photon prop
    def propagate(self):
        """Calculates energy losses, generates photon yields, and propagates photons"""
        if config["photon propagator"]["name"].lower()=="olympus":
            rstate = np.random.RandomState(config["run"]["random state seed"])
            rstate_jax = random.PRNGKey(config["run"]["random state seed"])
            # TODO this feels like it shouldn't be in the config
            config["photon propagator"]["olympus"]["runtime"] = {
                "random state": rstate,
                "random state jax": rstate_jax,
            }

        if config["run"]["subset"] is not None:
            nevents = config["run"]["subset"]
        else:
            nevents = len(self.injection)

        with tqdm(enumerate(self.injection), total=len(self.injection)) as pbar:
            for idx, injection_event in pbar:
                if idx == nevents:
                    break
                for final_state in injection_event.final_states:
                    pbar.set_description(f"Propagating {final_state}")
                    self._photon_propagator.propagate(final_state)
        if config["photon propagator"]["name"].lower()=="olympus":
            config["photon propagator"]["olympus"]["runtime"] = None

    def sim(self):
        """Performs injection of precipitating interaction, calculates energy losses,
        calculates photon yield, propagates photons, and save resultign photons"""
        if "runtime" in config["photon propagator"].keys():
            config["photon propagator"]["runtime"] = None
        #with open(config["general"]["config location"], "w") as f:
        #    json.dump(config, f, indent=2)
        self.inject()
        self.propagate()
        self.construct_output()
        if config["run"]["clean up"]:
            self._clean_up()

    def construct_output(self):
        """Constructs a parquet file with metadata from the generated files.
        Currently this still treats olympus and ppc output differently."""
        sim_switch = config["photon propagator"]["name"]

        print(f"Generating output for a {sim_switch} simulation.")
        print("Generating the different particle fields...")
        from .utils.serialization import serialize_particles_to_awkward, set_serialization_index
        set_serialization_index(self.injection)
        outarr = ak.Record({"config": config})
        photon_paths = config["photon propagator"][config["photon propagator"]["name"]]["paths"]
        #outarr = ak.with_field(outarr, ak.Record(config), where="config")
        outarr = ak.with_field(outarr, self.injection.to_awkward(), where="mc_truth")
        test_arr = serialize_particles_to_awkward(self.detector, self.injection)
        if test_arr is not None:
            outarr = ak.with_field(
                outarr,
                test_arr,
                where=photon_paths["photon field name"]
            )
        outfile = photon_paths['outfile']
        print(outfile)
        ak.to_parquet(outarr, outfile)
        table = pq.read_table(outfile)
        custom_meta_data = json.dumps(config)
        custom_meta_data_key = "config_prometheus"
        combined_meta = {custom_meta_data_key.encode() : custom_meta_data.encode()}
        table = table.replace_schema_metadata(combined_meta)
        pq.write_table(table, outfile)

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
                    f"{config['run']['storage location']}{key}.parquet"
                )
            except FileNotFoundError:
                continue
