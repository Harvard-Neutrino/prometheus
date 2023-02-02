# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,

import numpy as np
import proposal as pp

from .lepton_propagator import LeptonPropagator
from .loss import Loss
from ..particle import Particle, particle_from_proposal
# TODO This doesn't need to be here
from ..detector import Detector
from ..utils import iter_or_rep
from ..utils.units import GeV_to_MeV, MeV_to_GeV, cm_to_m, m_to_cm

def make_particle_def(particle: Particle) -> pp.particle.ParticleDef:
    """Makes a PROPOSAL particle definition

    params
    ______
    particle: Prometheus particle for which we want a PROPOSAL ParticleDef

    returns
    _______
    pdef: PROPOSAL particle definition
    """
    if str(particle) not in 'MuMinus MuPlus EMinus EPlus TauMinus TauPlus'.split():
        raise ValueError(f"Particle string {str(particle_string)} not recognized")
    pdef = getattr(pp.particle, f'{str(particle)}Def')()
    return pdef

def init_dynamic_data(
    particle: Particle,
    particle_definition: pp.particle.ParticleDef
) -> pp.particle.DynamicData:
    """Makes PROPOSAL DynamicData:

    params
    ______
    particle: Prometheus you want DynamicData for
    particle_definition: PROPOSAL particle definition

    returns
    _______
    particle_dd: PROPOSAL DynamicData for input particle
    """
    particle_dd = pp.particle.DynamicData(particle_definition.particle_type)
    particle_dd.position = particle.pp_position
    particle_dd.direction = particle.pp_direction
    particle_dd.energy = particle.e * GeV_to_MeV
    return particle_dd

def make_medium(medium_string: str) -> pp.medium.Medium:
    """
    Makes a proposal medium

    params
    ------
    medium_string: String which defines the medium in which
        proposal should propagate

    returns
    -------
    medium_def: Medium in which the propagation should take place
    """
    if medium_string.lower() not in 'water ice'.split():
        raise ValueError(f"Medium {medium_string} not supported at this time.")
    medium_def = getattr(pp.medium, medium_string.capitalize())()
    return medium_def

def init_sector(
    start: float,
    end: float,
    ecut: float,
    vcut: float
) -> pp.SectorDefinition:
    """Make a PROPOSAL spherical shell sector definition

    params
    ______
    start: inner radius of the shell (m)
    end: outer radius of the shell (m)
    ecut: Absolute energy cutoff below which to treat losses continuously
    vcut: Relative energy cutoff below which to treat losses continuously

    returns
    _______
    sec_def: PROPOSAL sector definition with spherical geometry
    """
    #Define a sector
    sec_def = pp.SectorDefinition()
    sec_def.geometry = pp.geometry.Sphere(
        pp.Vector3D(),
        end * m_to_cm,
        start * m_to_cm
    )
    sec_def.cut_settings.ecut = ecut * GeV_to_MeV
    sec_def.cut_settings.vcut = vcut
    # What should the default behavior of this be ?
    return sec_def

def make_propagator(
    particle: str,
    simulation_specs: dict,
    path_dict: dict
) -> pp.Propagator:
    """Make a PROPOSAL propagator

    params
    ______
    particle: Prometheus particle for which we want a PROPOSAL propagator
    simulation_specs: Dictionary specifying the configuration settings
    path_dict: Dictionary specifying any required path variables

    returns
    _______
    prop: PROPOSAL propagator for input Particle
    """
    pdef = make_particle_def(particle)
    inner_r = [0, simulation_specs["inner radius"]]
    outer_r = [simulation_specs["inner radius"], simulation_specs["maximum radius"]]
    ecut = iter_or_rep(simulation_specs["ecut"])
    vcut = iter_or_rep(simulation_specs["vcut"])
    medium = make_medium(simulation_specs["medium"])
    detector = pp.geometry.Sphere(
        pp.Vector3D(), simulation_specs["maximum radius"] * m_to_cm, 0.0
    )
    sec_defs = []
    for start, end, ecut, vcut in zip(inner_r, outer_r, ecut, vcut):
        sec_def = init_sector(start, end, ecut, vcut)
        sec_def.medium = medium
        sec_def.particle_location = pp.ParticleLocation.inside_detector
        sec_def.scattering_model = (
            getattr(pp.scattering.ScatteringModel, simulation_specs["scattering model"])
        )
	    # Bool options
        sec_def.crosssection_defs.brems_def.lpm_effect = simulation_specs["lpm effect"]
        sec_def.crosssection_defs.epair_def.lpm_effect = simulation_specs["lpm effect"]
        sec_def.do_continuous_randomization = simulation_specs["continuous randomization"]
        sec_def.do_continuous_energy_loss_output = simulation_specs['soft losses']
        sec_defs.append(sec_def)

    interpolation_def = pp.InterpolationDef()
    interpolation_def.path_to_tables = path_dict["tables path"]
    interpolation_def.path_to_tables_readonly = path_dict["tables path"]

    prop = pp.Propagator(
        pdef, sec_defs, detector, interpolation_def
    )
    return prop
    
def old_proposal_losses(
    prop: pp.Propagator,
    pdef: pp.particle.ParticleDef,
    particle: Particle,
    padding: float,
    # TODO should this be something that we do when serializing ?
    # I think the last three args of this function are just shit that make this
    # dumb thing work
    r_inice: float,
    detector_center: np.ndarray
) -> Particle:
    """Propagates charged lepton using PROPOSAL version <= 6

    params
    ______
    prop: PROPOSAL propagator object for the charged lepton to be propagated
    pdef: PROPOSAL particle definition for the charged lepton
    particle: Prometheus particle object to be propagated
    padding: Distance to propagate the charged lepton beyond its distance from the
        center of the detector
    r_inice: Distance from the center of the edge detector where losses should be
        recorded. This should be a few scattering lengths for accuracy, but not too
        much more because then you will propagate light which never makes it
    detector_center: Center of the detector in meters

    returns
    _______
    propped_particle: PROMETHEUS particle after propagation, including energy
        losses and children
    """
    particle_dd = init_dynamic_data(particle, pdef)
    propagation_length = np.linalg.norm(particle.position) + padding
    secondarys = prop.propagate(
        particle_dd, propagation_length * m_to_cm
    )
    del particle_dd
    continuous_loss = 0
    for sec in secondarys.particles:
        sec_energy = (sec.parent_particle_energy - sec.energy) * MeV_to_GeV
        # This should work with just the position but that requires some testing
        pos = np.array([sec.position.x, sec.position.y, sec.position.z]) * cm_to_m
        if sec.type > 1000000000: # This is an energy loss
            if np.linalg.norm(pos - detector_center) <= r_inice:
                particle.add_loss(Loss(sec.type, sec_energy, pos))
        else: # This is a particle. Might need to propagate
            child = particle_from_proposal(sec, parent=particle)
            particle.add_child(child)
    total_loss = None
    return particle

class OldProposalLeptonPropagator(LeptonPropagator):
    """Class for propagating charged leptons with PROPOSAL versions <= 6"""
    def __init__(self, config):
        super().__init__(config)

    def _make_propagator(self, particle: Particle) -> pp.Propagator:
        """Make a PROPOSAL propagator

        params
        ______
        particle: Prometheus Particle that you want a PROPOSAL propagator for

        returns
        _______
        propagator: PROPOSAL propagator
        """ 
        propagator = make_propagator(
            particle,
            self._config["simulation"],
            self._config["paths"]
        )
        return propagator

    def _make_particle_def(self, particle: Particle):
        """Make a PROPOSAL ParticleDef

        params
        ______
        particle: Prometheus Particle that you want a PROPOSAL ParticleDef for

        returns
        _______
        pdef: PROPOSAL ParticleDef
        """ 
        pdef = make_particle_def(particle)
        return pdef

    def energy_losses(
        self, 
        particle: Particle,
        detector: Detector
    ) -> Particle:
        """Propagate a particle and track the losses

        params
        ______
        particle: Prometheus Particle that should be propagated
        detector: Detector that this is being propagated within
            This is a temporary fix and will hopefully we solved soon :-)

        returns
        _______
        propped_particle: Prometheus Particle after propagation
        """
        particle_def, propagator = self[particle]
        propped_particle = old_proposal_losses(
            propagator,
            particle_def,
            particle,
            self._config["simulation"]["propagation padding"],
            detector.outer_radius + 1000.0,
            detector.offset
        )
        return propped_particle
