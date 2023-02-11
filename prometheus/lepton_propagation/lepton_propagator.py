from abc import abstractmethod
from typing import Tuple
import proposal as pp

from ..particle import Particle

class LeptonPropagator:
    '''Interface class to the different lepton propagators
    '''
    def __init__(self, config):
        self._prop_dict = {}
        self._pdef_dict = {}
        self._config = config

    def __getitem__(
        self,
        particle: Particle
    ) -> Tuple[pp.particle.ParticleDef, pp.Propagator]:
        """Retrieve the appropriate PROPOSAL ParticleDef and Propagator for a given 
        Prometheus Particle

        params
        ______
        particle: Prometheus Particle you want stuff for
        """
        if str(particle) not in self._pdef_dict.keys():
            self._pdef_dict[str(particle)] = self._make_particle_def(particle)
        if str(particle) not in self._prop_dict.keys():
            self._prop_dict[str(particle)] = self._make_propagator(particle)
        return self._pdef_dict[str(particle)], self._prop_dict[str(particle)]

    @property
    def config(self) -> dict:
        """Get the configuration dictionary used to make this"""
        return self._config
    
    @abstractmethod
    def _make_propagator(self, particle: Particle) -> pp.Propagator:
        """Makes a PROPOSAL Propagator
        params
        ______
        particle: Prometheus particle you want a particle def for
        """
        pass

    @abstractmethod
    def _make_particle_def(self, particle: Particle) -> pp.particle.ParticleDef:
        """Makes a PROPOSAL ParticleDef
        
        params
        ______
        particle: Prometheus particle you want a particle def for
        """
        pass

    @abstractmethod
    def energy_losses(self, particle: Particle) -> None:
        """Propagates particle with energy losses. The losses will be
            stored in `particle.losses`

        particle: Prometheus particle to propagate
        """
        pass
