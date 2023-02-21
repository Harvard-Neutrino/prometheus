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
from jax import random  # noqa: E402

from .utils import (
    config_mims, clean_config,
    UnknownInjectorError, UnknownLeptonPropagatorError,
    UnknownPhotonPropagatorError, NoInjectionError,
    InjectorNotImplementedError, CannotLoadDetectorError
)
from .config import config
from .detector import Detector
from .injection import RegisteredInjectors, INJECTION_CONSTRUCTOR_DICT
from .lepton_propagation import RegisteredLeptonPropagators
from .photon_propagation import (
    get_photon_propagator,
    RegisteredPhotonPropagators
)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

class PpcTmpdirExistsError(Exception):
    """Raised if PPC tmpdir exists and force not specified"""
    def __init__(self, path):
        self.message = f"{path} exists. Please remove it or specify force in the config"
        super().__init__(self.message)

def regularize(s: str) -> str:
    """Helper fnuction to regularize strings

    params
    ______
    s: string to regularize

    returns
    _______
    s: regularized string
    """
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
            If this is left out, the path from the `userconfig["detector"]["geo file"]`
            be loaded

        raises
        ______
        UnknownInjectorError: If we don't know how to handle the injector the config
            is asking for
        UnknownLeptonPropagatorError: If we don't know how to handle the lepton
            propagator you are asking for
        UnknownPhotonPropagatorError: If we don't know how to handle the photon
            propagator you are asking for
        CannotLoadDetectorError: When no detector provided and no
            geo file path provided in config

        """
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        if regularize(config["injection"]["name"]) not in RegisteredInjectors.list():
            raise UnknownInjectorError(config["injection"]["name"] + "is not supported as an injector!")

        if regularize(config["lepton propagator"]["name"]) not in RegisteredLeptonPropagators.list():
            raise UnknownLeptonPropagatorError(config["lepton propagator"]["name"] + "is not a known lepton propagator")

        if regularize(config["photon propagator"]["name"]) not in RegisteredPhotonPropagators.list():
            raise UnknownPhotonPropagatorError(config["photon propagator"]["name"] + " is not a known photon propagator")

        if detector is None and config["detector"]["geo file"] is None:
            raise CannotLoadDetectorError("No Detector provided and no geo file path given in config")

        if detector is None:
            from .detector import detector_from_geo
            detector = detector_from_geo(config["detector"]["geo file"])

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

        pp.RandomGenerator.get().set_seed(config["run"]["random state seed"])
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
            raise NoInjectionError("Injection has not been set!")
        return self._injection

    def inject(self):
        """Determines initial neutrino and final particle states according to config"""
        injection_config = config["injection"][config["injection"]["name"]]
        if injection_config["inject"]:

            from .injection import INJECTOR_DICT
            if self._injector not in INJECTOR_DICT.keys():
                raise InjectorNotImplementedError(str(self._injector) + " is not a registered injector" )

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
        elif config["photon propagator"]["name"].lower()=="ppc":
            from glob import glob
            import shutil
            from .utils.clean_ppc_tmpdir import clean_ppc_tmpdir
            if (
                os.path.exists(config['photon propagator']["PPC"]["paths"]["ppc_tmpdir"]) and \
                not config["photon propagator"]["PPC"]["paths"]["force"]
            ):
                raise PpcTmpdirExistsError(
                    config['photon propagator']["PPC"]["paths"]["ppc_tmpdir"]
                )
            os.mkdir(config['photon propagator']["PPC"]["paths"]["ppc_tmpdir"])
            fs = glob(f"{config['photon propagator']['PPC']['paths']['ppctables']}/*")
            for f in fs:
                shutil.copy(f, config['photon propagator']["PPC"]["paths"]["ppc_tmpdir"])
        elif config["photon propagator"]["name"].lower()=="ppc_cuda":
            from glob import glob
            import shutil
            from .utils.clean_ppc_tmpdir import clean_ppc_tmpdir
            if (
                os.path.exists(config['photon propagator']["PPC_CUDA"]["paths"]["ppc_tmpdir"]) and \
                not config["photon propagator"]["PPC_CUDA"]["paths"]["force"]
            ):
                raise PpcTmpdirExistsError(
                    config['photon propagator']["PPC_CUDA"]["paths"]["ppc_tmpdir"]
                )
            elif os.path.exists(config['photon propagator']["PPC_CUDA"]["paths"]["ppc_tmpdir"]):
                clean_ppc_tmpdir(config['photon propagator']["PPC_CUDA"]["paths"]["ppc_tmpdir"])
            os.mkdir(config['photon propagator']["PPC_CUDA"]["paths"]["ppc_tmpdir"])
            fs = glob(f"{config['photon propagator']['PPC_CUDA']['paths']['ppctables']}/*")
            for f in fs:
                shutil.copy(f, config['photon propagator']["PPC_CUDA"]["paths"]["ppc_tmpdir"])

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
        elif config["photon propagator"]["name"].lower()=="ppc":
            clean_ppc_tmpdir(config['photon propagator']['PPC']['paths']['ppc_tmpdir'])
        elif config["photon propagator"]["name"].lower()=="ppc_cuda":
            clean_ppc_tmpdir(config['photon propagator']['PPC_CUDA']['paths']['ppc_tmpdir'])


    def sim(self):
        """Performs injection of precipitating interaction, calculates energy losses,
        calculates photon yield, propagates photons, and save resultign photons"""
        if "runtime" in config["photon propagator"].keys():
            config["photon propagator"]["runtime"] = None
        self.inject()
        self.propagate()
        self.construct_output()

    def construct_output(self):
        """Constructs a parquet file with metadata from the generated files.
        Currently this still treats olympus and ppc output differently."""
        sim_switch = config["photon propagator"]["name"]

        from .utils.serialization import serialize_particles_to_awkward, set_serialization_index
        set_serialization_index(self.injection)
        json_config = json.dumps(config)
        # builder = ak.ArrayBuilder()
        # with builder.record('config'):
        #     builder.field('config').append(json_config)
        # outarr = builder.snapshot()
        # outarr = ak.Record({"config": json_config})
        # outarr['mc_truth'] = self.injection.to_awkward()
        test_arr = serialize_particles_to_awkward(self.detector, self.injection)
        if test_arr is not None:
            outarr = ak.Array({
                'mc_truth': self.injection.to_awkward(),
                config["photon propagator"]["photon field name"]: test_arr
            })
        else:
            outarr = ak.Array({
                'mc_truth': self.injection.to_awkward()
            })
        outfile = config["photon propagator"][config["photon propagator"]["name"]]["paths"]['outfile']
        # Converting to pyarrow table
        outarr = ak.to_arrow_table(outarr)
        custom_meta_data_key = "config_prometheus"
        combined_meta = {custom_meta_data_key.encode() : json_config.encode()}
        outarr = outarr.replace_schema_metadata(combined_meta)
        pq.write_table(outarr, outfile)

    def __del__(self):
        """What to do when the Prometheus instance is deleted
        """
        print("I am melting.... AHHHHHH!!!!")

