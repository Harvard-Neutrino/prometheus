from abc import abstractmethod
from typing import Tuple
import proposal as pp

from ..particle import Particle

class LeptonPropagator:
    """Interface class for the different lepton propagators."""
    def __init__(self, config):
        self._prop_dict = {}
        self._pdef_dict = {}
        self._config = config

    def __getitem__(
        self,
        particle: Particle
    ) -> Tuple[pp.particle.ParticleDef, pp.Propagator]:
        """Retrieve the PROPOSAL ``ParticleDef`` and ``Propagator`` for a particle.

        Parameters
        ----------
        particle : Particle
            Prometheus particle you want definitions for.

        Returns
        -------
        tuple
            Tuple of ``(ParticleDef, Propagator)`` corresponding to the input particle.
        """
        if str(particle) not in self._pdef_dict.keys():
            self._pdef_dict[str(particle)] = self._make_particle_def(particle)
        if str(particle) not in self._prop_dict.keys():
            self._prop_dict[str(particle)] = self._make_propagator(particle)
        return self._pdef_dict[str(particle)], self._prop_dict[str(particle)]

    @property
    def config(self) -> dict:
        """Get the configuration dictionary used to make this propagator."""
        return self._config
    
    @abstractmethod
    def _make_propagator(self, particle: Particle) -> pp.Propagator:
        """Create a PROPOSAL propagator for a Prometheus particle."""
        pass

    @abstractmethod
    def _make_particle_def(self, particle: Particle) -> pp.particle.ParticleDef:
        """Creates a PROPOSAL ``ParticleDef`` for a Prometheus particle."""
        pass

    @abstractmethod
    def energy_losses(self, particle: Particle) -> None:
        """Propagate particle with energy losses. The losses will be
            stored in ``particle.losses``.

        Parameters
        ----------
        particle : Particle
            Prometheus particle to propagate.
        """
        pass
