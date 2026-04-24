import warnings
from pathlib import Path

import numpy as np

from hyperion.constants import Constants
from hyperion.medium import medium_collections

from ..detector import Detector, Medium
from ..lepton_propagation import LeptonPropagator
from ..particle import Particle
from .hit import Hit
from .olympus.event_generation.event_generation import (
    generate_cascade,
    generate_realistic_track,
)
from .olympus.event_generation.lightyield import make_realistic_cascade_source
from .olympus.event_generation.photon_propagation.norm_flow_photons import (
    make_generate_norm_flow_photons,
)
from .olympus.event_generation.utils import sph_to_cart_jnp
from .photon_propagator import PhotonPropagator
from .registry import register_propagator

# Map Medium enum values to registered medium_collections keys.
# When a detector's medium has no dedicated model registered, the propagator
# falls back to the Cascadia Basin (P-ONE) water model and emits a warning.
_WATER_MEDIUM_MAP: dict = {
    Medium.WATER: "pone",
}

# Map medium_collections keys to (flow_filename, counts_filename) pairs.
# Add entries here when new model files are trained and committed to
# resources/olympus_resources/.  The user can always override both filenames
# explicitly via config.photon_propagator.olympus.paths.flow / .counts.
_FLOW_MODEL_MAP: dict[str, tuple[str, str]] = {
    "pone": (
        "photon_arrival_time_nflow_params.pickle",
        "photon_arrival_time_counts_params.pickle",
    ),
    # "antares": (
    #     "antares_nflow_params.pickle",
    #     "antares_counts_params.pickle",
    # ),
}

# Default filenames used to detect whether the user has overridden them.
_DEFAULT_FLOW_FILE = "photon_arrival_time_nflow_params.pickle"
_DEFAULT_COUNTS_FILE = "photon_arrival_time_counts_params.pickle"


@register_propagator("olympus")
class OlympusPhotonPropagator(PhotonPropagator):
    """Photon propagator that uses Olympus to propagate photons."""

    def __init__(self, lepton_propagator: LeptonPropagator, detector: Detector, config: dict):
        """Initialize the ``OlympusPhotonPropagator``.

        Parameters
        ----------
        lepton_propagator : LeptonPropagator
            Prometheus lepton propagator used to compute energy losses.
        detector : Detector
            Prometheus detector object.
        config : dict
            Olympus photon propagator configuration dictionary.
        """
        super().__init__(lepton_propagator, detector, config)

        if not self.config["simulation"]["files"]:
            ValueError("Currently only file runs for olympus are supported!")

        medium_key = _WATER_MEDIUM_MAP.get(detector.medium)
        if medium_key is None:
            warnings.warn(
                f"No dedicated optical model is registered for medium "
                f"'{detector.medium.name}'. Falling back to the Cascadia Basin "
                f"(P-ONE) water model. Simulation results may not accurately "
                f"reflect '{detector.medium.name}' optical properties.",
                UserWarning,
                stacklevel=2,
            )
            medium_key = "pone"
        self._ref_ix_f, self._sca_a_f, self._sca_l_f = medium_collections[medium_key]

        # Select flow / counts model files.  If the user has explicitly set
        # non-default filenames in their config those take precedence; otherwise
        # auto-select based on the resolved medium key, falling back to P-ONE
        # with a warning when no dedicated model is registered yet.
        cfg_flow = self.config["paths"]["flow"]
        cfg_counts = self.config["paths"]["counts"]
        if cfg_flow != _DEFAULT_FLOW_FILE or cfg_counts != _DEFAULT_COUNTS_FILE:
            # User override — use exactly what the config says.
            flow_file, counts_file = cfg_flow, cfg_counts
        elif medium_key in _FLOW_MODEL_MAP:
            flow_file, counts_file = _FLOW_MODEL_MAP[medium_key]
        else:
            warnings.warn(
                f"No dedicated flow model is registered for medium '{medium_key}'. "
                f"Falling back to the P-ONE model files. "
                f"Timing predictions may not accurately reflect '{medium_key}' optical properties.",
                UserWarning,
                stacklevel=2,
            )
            flow_file, counts_file = _FLOW_MODEL_MAP["pone"]

        location = self.config["paths"]["location"]
        # Ensure robust path joining whether `location` ends with a slash or not
        flow_path = str(Path(location) / flow_file)
        counts_path = str(Path(location) / counts_file)
        self._gen_ph = make_generate_norm_flow_photons(
            flow_path,
            counts_path,
            c_medium=self._c_medium_f(self.config["simulation"]["wavelength"]) / 1e9,
        )

    def propagate(self, particle: Particle, rng_key):
        """Simulate losses and propagate resulting photons for an input particle.

        Losses and resulting photon hits are stored within the input particle
        (which is modified in-place).

        Parameters
        ----------
        particle : Particle
            Prometheus particle object to simulate.
        rng_key : jax.random.PRNGKey
            JAX PRNG key for this particle's random draws.
        """

        # neutrinos don't produce light
        if abs(int(particle)) in [12, 14, 16]:
            return

        prop_distance = (
            np.linalg.norm(particle.position - self.detector.offset)
            + self.lepton_propagator.config["simulation"]["propagation padding"]
        )

        injection_event = {
            "time": 0.0,
            "theta": particle.theta,
            "phi": particle.phi,
            "pos": particle.position,
            "energy": particle.e,
            "particle_id": particle.pdg_code,
            "length": prop_distance,
            #'length': config['lepton propagator']['track length'],
        }
        event_dir = sph_to_cart_jnp(injection_event["theta"], injection_event["phi"])
        injection_event["dir"] = event_dir
        # Tracks
        if injection_event["particle_id"] in self.config["particles"]["track particles"]:
            _, proposal_prop = self.lepton_propagator[particle]
            res_event, _ = generate_realistic_track(
                self.detector,
                injection_event,
                key=rng_key,
                pprop_func=self._gen_ph,
                proposal_prop=proposal_prop,
                splitter=self.config["simulation"]["splitter"],
            )
        # Cascades
        else:
            import functools

            res_event, _ = generate_cascade(
                self.detector,
                injection_event,
                seed=rng_key,
                converter_func=functools.partial(
                    make_realistic_cascade_source, moliere_rand=True, resolution=0.2
                ),
                pprop_func=self._gen_ph,
                splitter=self.config["simulation"]["splitter"],
            )

        hits = []
        nstrings = len(set([mod.key[0] for mod in self.detector.modules]))
        string_idx = 0
        om_idx = 0
        oms_per_string = len(self.detector.modules) / nstrings
        for dom_hits in res_event:
            if om_idx == oms_per_string:
                om_idx = 0
                string_idx += 1
            for hit in dom_hits:
                hits.append(Hit(string_idx, om_idx, float(hit), None, None, None, None, None))
            om_idx += 1
        particle.hits = hits
        for child in particle.children:
            if child.e < 1:
                continue
            self.propagate(child, rng_key)

    def _c_medium_f(self, wl):
        """Speed of light in medium for a given wavelength in nm."""
        return Constants.BaseConstants.c_vac / self._ref_ix_f(wl)
