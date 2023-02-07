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
        photon_prop_config: dictionary containing all the configuration settings
            for photon propagation
        """
        self._lepton_propagator = lepton_propagator
        self._detector = detector
        self._config = photon_prop_config

    @abstractmethod
    def propagate(self, particle):
        pass

    @property
    def config(self) -> dict:
        return self._config

    @property
    def detector(self) -> Detector:
        return self._detector

    @property
    def lepton_propagator(self) -> LeptonPropagator:
        return self._lepton_propagator
