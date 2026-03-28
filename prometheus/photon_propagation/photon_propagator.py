from abc import abstractmethod

from ..lepton_propagation import LeptonPropagator
from ..detector import Detector

class PhotonPropagator:
    """Interface for handling different photon propagators."""
    def __init__(
        self,
        lepton_propagator: LeptonPropagator,
        detector: Detector,
        photon_prop_config: dict
    ):
        """Initialize the photon propagator.

        Parameters
        ----------
        lepton_propagator : LeptonPropagator
            Prometheus lepton propagator object which is used to generate losses from the particles.
        detector : Detector
            Prometheus detector object in which the light is propagated.
        photon_prop_config : dict
            Dictionary containing the configuration settings for photon propagation.
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
