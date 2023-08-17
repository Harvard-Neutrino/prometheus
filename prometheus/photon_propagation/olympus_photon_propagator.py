import numpy as np

from ..lepton_propagation import LeptonPropagator
from ..particle import Particle
from ..detector import Detector
from .photon_propagator import PhotonPropagator
from .hit import Hit

from olympus.event_generation.lightyield import make_realistic_cascade_source
from olympus.event_generation.utils import sph_to_cart_jnp

from olympus.event_generation.photon_propagation.norm_flow_photons import (
    make_generate_norm_flow_photons
)
from olympus.event_generation.event_generation import (
    generate_cascade,
    generate_realistic_track,
    simulate_noise,
)
from hyperion.medium import medium_collections
from hyperion.constants import Constants

class OlympusPhotonPropagator(PhotonPropagator):
    """PhotonPropagator the uses Olympus to propagate photons"""
    def __init__(
        self,
        lepton_propagator: LeptonPropagator,
        detector: Detector,
        config: dict
    ):
        super().__init__(lepton_propagator, detector, config)

        if not self.config['simulation']['files']:
            ValueError('Currently only file runs for olympus are supported!')

        #self._pprop_path = f"{self.config['location']}{self.config['photon model']}"
        # TODO This is reeeeeeeally bad I think
        medium_collections[detector.medium] = medium_collections["pone"]
        # The medium
        self._ref_ix_f, self._sca_a_f, self._sca_l_f = (
            medium_collections[detector.medium]
        )

        self._gen_ph = make_generate_norm_flow_photons(
            f"{self.config['paths']['location']}{self.config['paths']['flow']}",
            f"{self.config['paths']['location']}{self.config['paths']['counts']}",
            c_medium=self._c_medium_f(self.config['simulation']['wavelength']) / 1E9
        )

    def propagate(self, particle: Particle):
        """Simulate losses and propagate resulting photons for input particle

        params
        ______
        particle: Prometheus Particle object to simulate

        returns
        _______
        res_event: PLEASE FILL THIS IN
        res_record: PLEASE FILL THIS IN
        """

        # neutrinos don't produce light
        if abs(int(particle)) in [12, 14, 16]:
            return

        prop_distance = (
            np.linalg.norm(particle.position - self.detector.offset) 
            + self.lepton_propagator.config["simulation"]["propagation padding"]
        )

        injection_event = {
            "time": 0.,
            "theta": particle.theta,
            "phi": particle.phi,
            "pos": particle.position,
            "energy": particle.e,
            "particle_id": particle.pdg_code,
            'length': prop_distance,
            #'length': config['lepton propagator']['track length'],
        }
        event_dir = sph_to_cart_jnp(
            injection_event["theta"],
            injection_event["phi"]
        )
        injection_event["dir"] = event_dir
        # Tracks
        if injection_event['particle_id'] in self.config['particles']['track particles']:
            _, proposal_prop = self.lepton_propagator[particle]
            res_event, _ = (
                generate_realistic_track(
                    self.detector,
                    injection_event,
                    key=self.config['runtime']['random state jax'],
                    pprop_func=self._gen_ph,
                    proposal_prop=proposal_prop,
                    splitter=self.config['simulation']['splitter']
                )
            )
        # Cascades
        else:
            import functools
            res_event, _ = generate_cascade(
                self.detector,
                injection_event,
                seed = self.config['runtime']['random state jax'],
                converter_func = functools.partial(
                    make_realistic_cascade_source,
                    moliere_rand=True,
                    resolution=0.2),
                pprop_func=self._gen_ph,
                splitter=self.config['simulation']['splitter']
            )

        hits = []
        nstrings = len(set([mod.key[0] for mod in self.detector.modules]))
        string_idx = 0
        om_idx = 0
        oms_per_string = len(self.detector.modules) / nstrings
        for dom_hits in res_event:
            if om_idx==oms_per_string:
                om_idx = 0
                string_idx += 1
            for hit in dom_hits:
                hits.append(
                    Hit(string_idx, om_idx, float(hit), None,
                    None, None, None, None)
                )
            om_idx += 1
        particle.hits = hits
        for child in particle.children:
            if child.e < 1:
                continue
            self.propagate(child)


    def _c_medium_f(self, wl):
        """ Speed of light in medium for wl (nm)
        """
        return Constants.BaseConstants.c_vac / self._ref_ix_f(wl)
