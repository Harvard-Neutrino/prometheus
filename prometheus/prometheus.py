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

from .utils import (
    config_mims, clean_config,
    UnknownInjectorError,
    UnknownPhotonPropagatorError, NoInjectionError,
    InjectorNotImplementedError, CannotLoadDetectorError
)
from .config import config
from .detector import Detector
from .injection import RegisteredInjectors, INJECTION_CONSTRUCTOR_DICT
from .photon_propagation import (
    get_photon_propagator,
    RegisteredPhotonPropagators
)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

class PpcTmpdirExistsError(Exception):
    """Raised if ppc ``tmpdir`` exists and force not specified."""
    def __init__(self, path):
        self.message = f"{path} exists. Please remove it or specify force in the config"
        super().__init__(self.message)

def regularize(s: str) -> str:
    """Helper function to regularize strings.

    Parameters
    ----------
    s : str
        String to regularize.

    Returns
    -------
    s : str
        Regularized string.
    """
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.upper()
    return s


class Prometheus(object):
    """Class for unifying injection, energy loss calculation, and photon propagation."""
    def __init__(
        self,
        userconfig: Union[None, dict, str] = None,
        detector: Union[None, Detector] = None
    ) -> None:
        """Initialize the Prometheus class.

        Parameters
        ----------
        userconfig : dict or str or None
            Configuration dictionary or path to YAML file which specifies configuration.
        detector : Detector or None
            Detector to be used or path to geofile to load detector file.
            If this is left out, the path from the ``userconfig["detector"]["geo file"]`` will be loaded.

        Raises
        ------
        UnknownInjectorError
            Raised if we don't know how to handle the injector specified in the config.
        UnknownPhotonPropagatorError
            Raised if we don't know how to handle the photon
            propagator specified in the config.
        CannotLoadDetectorError
            Raised when no detector is provided and no geofile path is provided in config.
        """
        self._start_timing_misc = time()
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)


        if detector is None and config.detector.geo_file is None:
            raise CannotLoadDetectorError("No Detector provided and no geo file path given in config")

        if detector is None:
            from .detector import detector_from_geo
            detector = detector_from_geo(config.detector.geo_file)

        
        self._detector = detector
        self._injection = None

        # Infer which config to use from the PROPOSAL version
        # We need to check the version prior to import, otherwise
        # the type hinting will throw an error
        # We can probably hide this in MIMS
        import proposal as pp
        from .lepton_propagation.new_proposal_lepton_propagator import NewProposalLeptonPropagator as LeptonPropagator
        config.lepton_propagator.name = "new proposal"
        config.lepton_propagator.version = pp.__version__

        config_mims(config, self.detector)
        clean_config(config)

        self._injector = getattr(
            RegisteredInjectors,
            regularize(config.injection.name)
        )

        self._pp = getattr(
            RegisteredPhotonPropagators,
            regularize(config.photon_propagator.name)
        )

        if regularize(config.injection.name) not in RegisteredInjectors.list():
            raise UnknownInjectorError(config.injection.name + "is not supported as an injector!")

        if regularize(config.photon_propagator.name) not in RegisteredPhotonPropagators.list():
            raise UnknownPhotonPropagatorError(config.photon_propagator.name + " is not a known photon propagator")

        pp.RandomGenerator.get().set_seed(config.run.random_state_seed)
        lepton_prop_config = config.lepton_propagator[config.lepton_propagator.name]
        self._lepton_propagator = LeptonPropagator(lepton_prop_config)

        pp_config = config.photon_propagator[config.photon_propagator.name]
        self._photon_propagator = get_photon_propagator(config.photon_propagator.name)(
            self._lepton_propagator,
            self.detector,
            pp_config
        )
        self._end_timing_misc = time()


    @property
    def detector(self):
        return self._detector

    @property
    def injection(self):
        #if self._injection is None:
        #    raise NoInjectionError("Injection has not been set!")
        return self._injection

    def inject(self):
        """Determine initial neutrino and final particle states according to config."""
        injection_config = config.injection[config.injection.name]
        if injection_config.inject:

            from .injection import INJECTOR_DICT
            if self._injector not in INJECTOR_DICT.keys():
                raise InjectorNotImplementedError(str(self._injector) + " is not a registered injector" )

            injection_config.simulation.random_state_seed = config.run.random_state_seed
            INJECTOR_DICT[self._injector](
                injection_config.paths,
                injection_config.simulation,
                detector_offset=self.detector.offset
            )
        self._injection = INJECTION_CONSTRUCTOR_DICT[self._injector](
            injection_config.paths.injection_file
        )

    # We should factor out generating losses and photon prop
    def propagate(self):
        """Calculate energy losses, generate photon yields, and propagate photons."""
        pp_name = config.photon_propagator.name.lower()
        if pp_name == "olympus":
            rng_key = random.PRNGKey(config.run.random_state_seed)
        elif pp_name == "ppc":
            from glob import glob
            import shutil
            from .utils.clean_ppc_tmpdir import clean_ppc_tmpdir
            if (
                os.path.exists(config.photon_propagator.ppc.paths.ppc_tmpdir) and
                not config.photon_propagator.ppc.paths.force
            ):
                raise PpcTmpdirExistsError(config.photon_propagator.ppc.paths.ppc_tmpdir)
            os.mkdir(config.photon_propagator.ppc.paths.ppc_tmpdir)
            fs = glob(f"{config.photon_propagator.ppc.paths.ppctables}/*")
            for f in fs:
                shutil.copy(f, config.photon_propagator.ppc.paths.ppc_tmpdir)
        elif pp_name == "ppc_cuda":
            from glob import glob
            import shutil
            from .utils.clean_ppc_tmpdir import clean_ppc_tmpdir
            if (
                os.path.exists(config.photon_propagator.ppc_cuda.paths.ppc_tmpdir) and
                not config.photon_propagator.ppc_cuda.paths.force
            ):
                raise PpcTmpdirExistsError(config.photon_propagator.ppc_cuda.paths.ppc_tmpdir)
            elif os.path.exists(config.photon_propagator.ppc_cuda.paths.ppc_tmpdir):
                clean_ppc_tmpdir(config.photon_propagator.ppc_cuda.paths.ppc_tmpdir)
            os.mkdir(config.photon_propagator.ppc_cuda.paths.ppc_tmpdir)
            fs = glob(f"{config.photon_propagator.ppc_cuda.paths.ppctables}/*")
            for f in fs:
                shutil.copy(f, config.photon_propagator.ppc_cuda.paths.ppc_tmpdir)

        nevents = len(self.injection)

        with tqdm(enumerate(self.injection), total=len(self.injection)) as pbar:
            for idx, injection_event in pbar:
                if idx == nevents:
                    break
                for final_state in injection_event.final_states:
                    pbar.set_description(f"Propagating {final_state}")
                    if pp_name == "olympus":
                        rng_key, subkey = random.split(rng_key)
                    else:
                        subkey = None
                    self._photon_propagator.propagate(final_state, subkey)
        if pp_name == "olympus":
            pass
        elif pp_name == "ppc":
            clean_ppc_tmpdir(config.photon_propagator.ppc.paths.ppc_tmpdir)
        elif pp_name == "ppc_cuda":
            clean_ppc_tmpdir(config.photon_propagator.ppc_cuda.paths.ppc_tmpdir)


    def sim(self):
        """Perform injection of precipitating interaction, calculate energy losses, calculate photon yield, propagate photons, and save resulting photons."""
        start_inj = time()
        self.inject()
        end_inj = time()
        start_prop = time()
        self.propagate()
        end_prop = time()
        start_out = time()
        self.construct_output()
        end_out = time()
        # Timing stuff
        # TODO: remove this?
        self._timing_arr = np.array([
            self._end_timing_misc - self._start_timing_misc,
            end_inj - start_inj,
            end_prop - start_prop,
            end_out - start_out,
        ])

    def construct_output(self):
        """Construct a parquet file with metadata from the generated files.
        
        Currently this still treats olympus and ppc output differently.
        """
        # sim_switch = config["photon propagator"]["name"]

        from .utils.serialization import serialize_particles_to_awkward, set_serialization_index
        set_serialization_index(self.injection)
        json_config = json.dumps(config.to_dict())
        test_arr = serialize_particles_to_awkward(self.detector, self.injection)
        if test_arr is not None:
            outarr = ak.Array({
                'mc_truth': self.injection.to_awkward(),
                config.photon_propagator.photon_field_name: test_arr
            })
        else:
            outarr = ak.Array({
                'mc_truth': self.injection.to_awkward()
            })
        outfile = config.run.outfile
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

