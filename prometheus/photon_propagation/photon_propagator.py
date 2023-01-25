from abc import abstractmethod

from ..lepton_propagation import LeptonPropagator
from ..detector import Detector

class PhotonPropagator:
    """Interface for handling different photon propagators"""
    def __init__(
        self,
        lepton_propagator: LeptonPropagator,
        detector: Detector,
        photon_prop_config: dict
    ):
        """Initialize the PhotonPropagator object
        
        params
        ______
        lepton_propagator: Prometheus LeptonPropagator object which will be used
            to generate losses from the particles
        detector: Prometheus detector object in which the light will be
            propagated
        """
        self._lepton_propagator = lepton_propagator
        self._detector = detector
        self._config = photon_prop_config

    @abstractmethod
    def propagate(self, particle):
        pass

    @property
    def config(self):
        return self._config

    @property
    def detector(self):
        return self._detector

    @property
    def lepton_propagator(self):
        return self._lepton_propagator
