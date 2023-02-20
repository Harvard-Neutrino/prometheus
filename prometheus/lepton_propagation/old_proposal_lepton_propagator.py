# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,

import numpy as np
import proposal as pp
from typing import List

from .lepton_propagator import LeptonPropagator
from .loss import Loss
from ..particle import Particle, particle_from_proposal
from ..detector import Detector
from ..utils.units import GeV_to_MeV, MeV_to_GeV, cm_to_m, m_to_cm

MEDIUM_DICT = {
    "INNERCORE": pp.medium.StandardRock,
    "OUTERCORE": pp.medium.StandardRock,
    "MANTLE": pp.medium.StandardRock,
    "ROCK": pp.medium.StandardRock,
    "ICE": pp.medium.Ice,
    "AIR": pp.medium.Air,
    "WATER": pp.medium.Water
}
def remove_comments(s: str) -> str:
    """Helper for removing trailing comments

    params
    ______
    s: string you want to remove comments from

    returns
    _______
    s: string without the comments
    """
    if "#" not in s:
        return s
    idx = s.index("#")
    return s[:idx]

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
        raise ValueError(f"Cannot propagate {str(particle)} with PROPOSAL")
    pdef = getattr(pp.particle, f'{str(particle)}Def')()
    return pdef

def make_detector(earth_file: str) -> pp.geometry.Sphere:
    """Make a PROPOSAL sphere with radius eqaul to the max radius
    from an Earth data file

    params
    ______
    earth_file: Earth datafile
    
    returns
    _______
    detector: PROPOSAL Sphere
    """
    with open(earth_file, "r") as f:
        for line in f:
            if line[0]=="#" or line[0]==" " or line[:1]=="\n":
                continue
            line = remove_comments(line)
            split_line = [x for x in line.replace("\n", "").split(" ") if len(x)>0]
            outer_radius = float(split_line[0])
    detector = pp.geometry.Sphere(pp.Vector3D(0,0,1), outer_radius, 0.0)
    return detector

def make_sector_defs(earth_file: str, simulation_specs: dict) -> List[pp.SectorDefinition]:
    """Make list of PROPOSAL sector definitions for an input Earth 
    data file according to given simulation specifications

    params
    ______
    earth_file: Earth datafile
    simulation_specs: dictionary specifying simulation paramters

    returns
    _______
    sec_defs: list of PROPOSAL sector definitions
    """
    inner_radius = 0
    sec_defs = []
    with open(earth_file, "r") as f:
        for line in f:
            if line[0]=="#" or line[0]==" " or line[:1]=="\n":
                continue
            line = remove_comments(line)
            split_line = [x for x in line.replace("\n", "").split(" ") if len(x)>0]
            outer_radius = float(split_line[0])
            # Compute average density in linear approx
            if len(split_line[4:])==1:
                rho_bar = float(split_line[4])
            else:
                p0 = float(split_line[4])
                p1 = float(split_line[5])
                rho_bar = p0 + p1 * (inner_radius + outer_radius) /2

            sector_def = pp.SectorDefinition()
            sector_def.cut_settings.ecut = simulation_specs["ecut"] * GeV_to_MeV
            sector_def.cut_settings.vcut = simulation_specs["vcut"]

            sector_def.geometry = pp.geometry.Sphere(
                pp.Vector3D(0,0,1),
                # Do not apply units. PROPOSAL has a bug that this needs meters not cm
                outer_radius,
                inner_radius,
                #outer_radius * m_to_cm,
                #inner_radius * m_to_cm,
            )
            test_medium = MEDIUM_DICT[split_line[2]]()
            medium = MEDIUM_DICT[split_line[2]](rho_bar / test_medium.mass_density)
            sector_def.medium = medium

            sector_def.crosssection_defs.brems_def.lpm_effect = simulation_specs["lpm effect"]
            sector_def.crosssection_defs.epair_def.lpm_effect = simulation_specs["lpm effect"]
            sector_def.do_continuous_randomization = simulation_specs["continuous randomization"]
            sector_def.do_continuous_energy_loss_output = simulation_specs['soft losses']

            sec_defs.append(sector_def)
            inner_radius = outer_radius
    return sec_defs

def init_dynamic_data(
    particle: Particle,
    particle_definition: pp.particle.ParticleDef,
    coordinate_shift: np.ndarray
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
    particle_dd.position = pp.Vector3D(*(particle.position + coordinate_shift) * m_to_cm)
    particle_dd.direction = pp.Vector3D(*particle.direction)
    particle_dd.energy = particle.e * GeV_to_MeV
    return particle_dd

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
    detector = make_detector(path_dict["earth model location"])
    sec_defs = make_sector_defs(path_dict["earth model location"], simulation_specs)
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
    detector_center: np.ndarray,
    coordinate_shift: np.ndarray
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
    """
    particle_dd = init_dynamic_data(particle, pdef, coordinate_shift)
    propagation_length = np.linalg.norm(particle.position) + padding
    secondarys = prop.propagate(
        particle_dd, propagation_length * m_to_cm
    )
    del particle_dd
    continuous_loss = 0
    for sec in secondarys.particles:
        sec_energy = (sec.parent_particle_energy - sec.energy) * MeV_to_GeV
        pos = np.array([sec.position.x, sec.position.y, sec.position.z]) * cm_to_m - coordinate_shift
        if sec.type > 1000000000: # This is an energy loss
            if np.linalg.norm(pos - detector_center) <= r_inice:
                particle.losses.append(Loss(sec.type, sec_energy, pos))
        else: # This is a particle. Might need to propagate
            child = particle_from_proposal(sec, coordinate_shift, parent=particle)
            particle.children.append(child)
    total_loss = None



class OldProposalLeptonPropagator(LeptonPropagator):
    """Class for propagating charged leptons with PROPOSAL versions <= 6"""
    def __init__(self, config: dict):
        with open(config["paths"]["earth model location"], "r") as f:
            for line in f:
                if line[0]=="#" or line[0]==" " or line[:1]=="\n":
                    continue
                line = remove_comments(line)
                split_line = [x for x in line.replace("\n", "").split(" ") if len(x)>0]
                if split_line[2]=="AIR":
                    break
                outer_radius = float(split_line[0]) 

        self._coordinate_shift = np.array([0, 0, outer_radius])
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
    ) -> None:
        """Propagate a particle and track the losses. Losses and 
        children are applied in place

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
        old_proposal_losses(
            propagator,
            particle_def,
            particle,
            self._config["simulation"]["propagation padding"],
            detector.outer_radius + 1000.0,
            detector.offset,
            self._coordinate_shift
        )
