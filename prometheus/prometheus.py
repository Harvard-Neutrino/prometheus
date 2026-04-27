# -*- coding: utf-8 -*-
# prometheus.py
# Copyright (C) 2022 Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the package

import json
import logging
import os
import warnings
from pathlib import Path
from time import time
from typing import Union

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from .config import config
from .detector import Detector
from .injection import INJECTION_CONSTRUCTOR_DICT, RegisteredInjectors
from .logging.handlers import LogCounterHandler
from .logging_config import configure_logging
from .photon_propagation import (
    RegisteredPhotonPropagators,
    get_propagator,
)
from .summary import emit_run_summary
from .utils import (
    CannotLoadDetectorError,
    InjectorNotImplementedError,
    UnknownInjectorError,
    UnknownPhotonPropagatorError,
    clean_config,
    config_mims,
)
from .utils.capture import _COutputCapture
from .utils.timing import time_block

# Legacy alias used in this file.
get_photon_propagator = get_propagator

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

logger = logging.getLogger(__name__)


class PpcTmpdirExistsError(Exception):
    """Raised if ppc ``tmpdir`` exists and force not specified.

    Parameters
    ----------
    path : str
        Path to the tmpdir.
    """

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
    str
        Regularized string.
    """
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.upper()
    return s


class Prometheus(object):
    """Class for unifying injection, energy loss calculation, and photon propagation.
    """

    def __init__(
        self, userconfig: Union[None, dict, str] = None, detector: Union[None, Detector] = None
    ) -> None:
        """Initialize the Prometheus class.

        Parameters
        ----------
        userconfig : dict or str or None
            Configuration dictionary or path to YAML file specifying configuration.
        detector : Detector or None
            Detector to be used or path to geo file to load detector from. If omitted,
            the path from ``userconfig["detector"]["geo file"]`` will be loaded.

        Raises
        ------
        UnknownInjectorError
            If the injector specified in the config is unknown.
        UnknownPhotonPropagatorError
            If the photon propagator specified in the config is unknown.
        CannotLoadDetectorError
            If no detector is provided and no geo file path is provided in config.
        """
        self._start_timing_misc = time()
        if userconfig is not None:
            from .config_types import PrometheusConfig

            if isinstance(userconfig, PrometheusConfig):
                # Caller passed a pre-built config object; use it directly.
                config.__dict__.update(userconfig.__dict__)
            elif isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        if detector is None and config.detector.geo_file is None:
            raise CannotLoadDetectorError(
                "No Detector provided and no geo file path given in config"
            )

        # Defer detector construction until after logging/warning capture is
        # configured below so we can suppress its noisy prints in user mode.
        self._detector = detector
        self._injection = None

        # Configure logging centrally from the config as early as possible
        try:
            configure_logging(config)
        except Exception:
            # Fallback to basic config if configuration fails
            logging.basicConfig(level=logging.WARNING)

        # Determine summary mode before capturing warnings/redirects
        try:
            summary_mode = getattr(config.run, "summary_mode", "user") or "user"
        except Exception:
            summary_mode = "user"
        self._summary_mode = summary_mode

        # Capture Python warnings into an in-memory list so we can present
        # them in a tidy summary rather than interleaving them with stdout.
        try:
            self._captured_warnings = []
            self._orig_showwarning = warnings.showwarning

            def _capture_warning(message, category, filename, lineno, file=None, line=None):
                try:
                    self._captured_warnings.append(str(message))
                except Exception:
                    pass
                # In debug mode, still show the warning immediately
                if getattr(self, "_summary_mode", "user") == "debug":
                    try:
                        self._orig_showwarning(message, category, filename, lineno, file, line)
                    except Exception:
                        pass

            warnings.showwarning = _capture_warning
        except Exception:
            self._captured_warnings = []

        # Attach a lightweight log-counter to capture warning/error counts
        try:
            self._log_counter = LogCounterHandler()
            logging.getLogger().addHandler(self._log_counter)
        except Exception:
            self._log_counter = None

        # Welcome banner (ascii logo) - show only if enabled in config.run.banner
        try:
            show_banner = False
            try:
                show_banner = bool(
                    getattr(config, "run", None) and getattr(config.run, "banner", False)
                )
            except Exception:
                show_banner = False
            logo_path = Path(__file__).resolve().parent.parent / "assets" / "ascii-logo.txt"
            if show_banner:
                try:
                    with open(logo_path, "r") as _fh:
                        logo = _fh.read()
                    if self._summary_mode == "user":
                        print("\n" + logo)
                    else:
                        logger.info("\n%s", logo)
                except Exception:
                    if show_banner and self._summary_mode != "user":
                        logger.debug("Could not load ascii logo from %s", logo_path)
        except Exception:
            pass

        # Construct detector after logging/warning capture is in place so
        # we can collapse any noisy prints from detector construction.
        try:
            if self._detector is None:
                from .detector import detector_from_geo

                if getattr(self, "_summary_mode", "user") == "user":
                    with _COutputCapture() as _cap:
                        self._detector = detector_from_geo(config.detector.geo_file)
                    try:
                        self._init_output = (
                            (getattr(self, "_init_output", "") or "")
                            + (_cap.out or "")
                            + (_cap.err or "")
                        )
                    except Exception:
                        self._init_output = getattr(self, "_init_output", "") or ""
                else:
                    self._detector = detector_from_geo(config.detector.geo_file)
        except Exception:
            # If detector fails to construct, re-raise to preserve behaviour
            raise

        # Now import PROPOSAL and the lepton propagator (kept after logging
        # configuration and warning capture so we can suppress their noise).
        import proposal as pp

        from .lepton_propagation.new_proposal_lepton_propagator import (
            NewProposalLeptonPropagator as LeptonPropagator,
        )

        config.lepton_propagator.name = "new proposal"
        config.lepton_propagator.version = pp.__version__

        config_mims(config, self.detector)
        clean_config(config)

        self._injector = getattr(RegisteredInjectors, regularize(config.injection.name))

        self._pp = getattr(RegisteredPhotonPropagators, regularize(config.photon_propagator.name))

        if regularize(config.injection.name) not in RegisteredInjectors.list():
            raise UnknownInjectorError(config.injection.name + "is not supported as an injector!")

        if regularize(config.photon_propagator.name) not in RegisteredPhotonPropagators.list():
            raise UnknownPhotonPropagatorError(
                config.photon_propagator.name + " is not a known photon propagator"
            )

        pp.RandomGenerator.get().set_seed(config.run.random_state_seed)
        lepton_prop_config = config.lepton_propagator[config.lepton_propagator.name]
        # Construct lepton propagator and photon propagator; these can be
        # noisy (print() calls) in some third-party components, so capture
        # their stdout/stderr in user mode and present it only in the
        # collapsed warnings section.
        try:
            if getattr(self, "_summary_mode", "user") == "user":
                with _COutputCapture() as _cap:
                    self._lepton_propagator = LeptonPropagator(lepton_prop_config)
                    pp_config = config.photon_propagator[config.photon_propagator.name]
                    self._photon_propagator = get_photon_propagator(config.photon_propagator.name)(
                        self._lepton_propagator, self.detector, pp_config
                    )
                try:
                    self._init_output = (_cap.out or "") + (_cap.err or "")
                except Exception:
                    self._init_output = ""
            else:
                self._lepton_propagator = LeptonPropagator(lepton_prop_config)
                pp_config = config.photon_propagator[config.photon_propagator.name]
                self._photon_propagator = get_photon_propagator(config.photon_propagator.name)(
                    self._lepton_propagator, self.detector, pp_config
                )
        except Exception:
            # If constructor fails, propagate the exception as before.
            raise
        self._end_timing_misc = time()

        # High-level initialization summary (print minimal in user mode so
        # it appears even when the root logger is elevated to WARNING).
        init_msg = (
            f"Prometheus initialized: run={config.run.run_number} nevents={config.run.nevents} "
            f"injector={config.injection.name} propagator={config.photon_propagator.name} "
            f"modules={getattr(self.detector, 'n_modules', len(getattr(self.detector, 'modules', [])))}"  # noqa: E501
        )
        if getattr(self, "_summary_mode", "user") == "user":
            try:
                print(init_msg)
            except Exception:
                logger.info("Prometheus initialized")
        else:
            logger.info(init_msg)
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug("Resolved config: %s", config.to_dict())
            except Exception:
                logger.debug("Resolved config: <unserializable>")

    @property
    def detector(self):
        return self._detector

    @property
    def injection(self):
        # if self._injection is None:
        #    raise NoInjectionError("Injection has not been set!")
        return self._injection

    def inject(self):
        """Determine initial neutrino and final particle states according to config.
        """
        injection_config = config.injection[config.injection.name]
        logger.info(
            "Starting injection: mode=%s inject=%s",
            config.injection.name,
            injection_config.inject,
        )
        with time_block("injection", logger):
            if injection_config.inject:
                from .injection import INJECTOR_DICT

                if self._injector not in INJECTOR_DICT.keys():
                    raise InjectorNotImplementedError(
                        str(self._injector) + " is not a registered injector"
                    )

                injection_config.simulation.random_state_seed = config.run.random_state_seed
                INJECTOR_DICT[self._injector](
                    injection_config.paths,
                    injection_config.simulation,
                    detector_offset=self.detector.offset,
                )
            try:
                self._injection = INJECTION_CONSTRUCTOR_DICT[self._injector](
                    injection_config.paths.injection_file
                )
            except Exception:
                logger.exception(
                    "Failed to construct injection from %s", injection_config.paths.injection_file
                )
                raise
        try:
            n_inj = len(self._injection)
        except Exception:
            n_inj = None
        logger.info(
            "Injection complete: loaded %s events from %s",
            n_inj,
            injection_config.paths.injection_file,
        )

    # We should factor out generating losses and photon prop
    def propagate(self, capture: bool = False):
        """Calculate energy losses, generate photon yields, and propagate photons.

        Parameters
        ----------
        capture : bool
            If ``True``, capture stdout/stderr emitted by downstream modules into
            `self._propagate_output` instead of letting it print to the console.
        """
        pp_name = config.photon_propagator.name.lower()
        logger.info("Starting propagation: propagator=%s", config.photon_propagator.name)
        try:
            total_events = len(self.injection)
        except Exception:
            total_events = None
        if total_events is not None:
            logger.info("Propagating %d events", total_events)
        if pp_name == "olympus":
            # Import lazily so JAX messages can be controlled by configure_logging
            from jax import random

            rng_key = random.PRNGKey(config.run.random_state_seed)
        elif pp_name == "ppc":
            import shutil
            from glob import glob

            from .utils.clean_ppc_tmpdir import clean_ppc_tmpdir

            ppc_tmpdir = Path(config.photon_propagator.ppc.paths.ppc_tmpdir)
            if ppc_tmpdir.exists() and not config.photon_propagator.ppc.paths.force:
                raise PpcTmpdirExistsError(str(ppc_tmpdir))
            ppc_tmpdir.mkdir(parents=True, exist_ok=False)
            fs = glob(str(Path(config.photon_propagator.ppc.paths.ppctables) / "*"))
            for f in fs:
                shutil.copy(f, str(ppc_tmpdir))
        elif pp_name == "ppc_cuda":
            import shutil
            from glob import glob

            from .utils.clean_ppc_tmpdir import clean_ppc_tmpdir

            ppc_cuda_tmpdir = Path(config.photon_propagator.ppc_cuda.paths.ppc_tmpdir)
            if ppc_cuda_tmpdir.exists() and not config.photon_propagator.ppc_cuda.paths.force:
                raise PpcTmpdirExistsError(str(ppc_cuda_tmpdir))
            elif ppc_cuda_tmpdir.exists():
                clean_ppc_tmpdir(str(ppc_cuda_tmpdir))
            ppc_cuda_tmpdir.mkdir(parents=True, exist_ok=False)
            fs = glob(str(Path(config.photon_propagator.ppc_cuda.paths.ppctables) / "*"))
            for f in fs:
                shutil.copy(f, str(ppc_cuda_tmpdir))

        nevents = len(self.injection)

        # Decide whether to show progress bars: only when number of events
        # exceed the configured `progress_threshold`.
        threshold = getattr(config.run, "progress_threshold", 10)
        show_progress = bool(nevents > threshold)

        # If caller requested capture but progress is meaningful, we avoid
        # capturing to allow interactive feedback.
        should_capture = bool(capture and not show_progress)

        cap = None
        if should_capture:
            cap = _COutputCapture()

        if cap is not None:
            cap.__enter__()
        try:
            if show_progress:
                iterator = tqdm(enumerate(self.injection), total=len(self.injection))
            else:
                iterator = enumerate(self.injection)

            for idx, injection_event in iterator:
                if idx == nevents:
                    break
                for final_state in injection_event.final_states:
                    if show_progress:
                        try:
                            iterator.set_description(f"Propagating {final_state}")
                        except Exception:
                            pass
                    if pp_name == "olympus":
                        # random already imported above for olympus
                        rng_key, subkey = random.split(rng_key)
                    else:
                        subkey = None
                    try:
                        self._photon_propagator.propagate(final_state, subkey)
                    except Exception:
                        logger.exception(
                            "Error propagating event %s final_state=%s", idx, final_state
                        )
                        raise
        finally:
            if cap is not None:
                cap.__exit__(None, None, None)
                try:
                    self._propagate_output = (cap.out or "") + (cap.err or "")
                except Exception:
                    self._propagate_output = ""
        logger.info("Propagation complete")
        if pp_name == "olympus":
            pass
        elif pp_name == "ppc":
            clean_ppc_tmpdir(config.photon_propagator.ppc.paths.ppc_tmpdir)
        elif pp_name == "ppc_cuda":
            clean_ppc_tmpdir(config.photon_propagator.ppc_cuda.paths.ppc_tmpdir)

    def sim(self):
        """Perform injection, calculate energy losses and photon yield, propagate photons,
        and save resulting photons.
        """
        logger.info("Starting full simulation run %s", config.run.run_number)
        # Record phase start/end timestamps on self so construct_output can summarise
        self._run_start_time = time()

        # Injection: capture printed noise in user mode so the console stays clean
        self._start_inj = time()
        try:
            if getattr(self, "_summary_mode", "user") == "user":
                with _COutputCapture() as _cap:
                    self.inject()
                try:
                    self._inject_output = (_cap.out or "") + (_cap.err or "")
                except Exception:
                    self._inject_output = ""
            else:
                self.inject()
        finally:
            self._end_inj = time()

        # Decide progress behaviour and whether to capture propagation output
        try:
            nevents = len(self._injection)
        except Exception:
            nevents = getattr(config.run, "nevents", 0)
        threshold = getattr(config.run, "progress_threshold", 10)
        show_progress = bool(nevents > threshold)
        capture_prop = (getattr(self, "_summary_mode", "user") == "user") and (not show_progress)

        self._start_prop = time()
        self.propagate(capture=capture_prop)
        self._end_prop = time()

        self._start_out = time()
        # construct_output will set self._end_out when finished
        self.construct_output()
        self._end_out = getattr(self, "_end_out", time())
        logger.info("Simulation run complete")
        # Timing array: misc, inj, prop, out
        self._timing_arr = np.array(
            [
                self._end_timing_misc - self._start_timing_misc,
                self._end_inj - self._start_inj,
                self._end_prop - self._start_prop,
                self._end_out - self._start_out,
            ]
        )
        try:
            logger.info(
                "Timings (s): misc=%.3f inj=%.3f prop=%.3f out=%.3f",
                float(self._timing_arr[0]),
                float(self._timing_arr[1]),
                float(self._timing_arr[2]),
                float(self._timing_arr[3]),
            )
        except Exception:
            pass

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
            outarr = ak.Array(
                {
                    "mc_truth": self.injection.to_awkward(),
                    config.photon_propagator.photon_field_name: test_arr,
                }
            )
        else:
            outarr = ak.Array({"mc_truth": self.injection.to_awkward()})
        outfile = config.run.outfile
        # Converting to pyarrow table
        outarr = ak.to_arrow_table(outarr)
        custom_meta_data_key = "config_prometheus"
        combined_meta = {custom_meta_data_key.encode(): json_config.encode()}
        outarr = outarr.replace_schema_metadata(combined_meta)
        with time_block("write_output", logger):
            pq.write_table(outarr, outfile)
        # Record end of write phase
        end_out = time()
        self._end_out = end_out

        size = None
        try:
            size = Path(outfile).stat().st_size
            logger.info("Wrote output to %s (%d bytes)", outfile, size)
        except Exception:
            logger.info("Wrote output to %s", outfile)

        # Build and emit run summary (extracted to prometheus.summary.emit_run_summary)
        try:
            emit_run_summary(self, outfile, end_out, size=size)
        except Exception:
            logger.debug("Failed to produce run summary")

    def __del__(self):
        """What to do when the Prometheus instance is deleted"""
        # Restore original warning handler if we replaced it
        try:
            if (
                hasattr(self, "_orig_showwarning")
                and getattr(warnings, "showwarning", None) is not None
            ):
                warnings.showwarning = self._orig_showwarning
        except Exception:
            pass
        # Avoid using logging during interpreter shutdown in __del__.
