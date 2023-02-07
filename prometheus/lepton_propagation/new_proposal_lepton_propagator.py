# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,

import numpy as np
import proposal as pp
from typing import List

from ..particle import Particle, particle_from_proposal
from ..detector import Detector
from ..utils.units import GeV_to_MeV, MeV_to_GeV, cm_to_m, m_to_cm
from .loss import Loss
from .lepton_propagator import LeptonPropagator

MEDIUM_DICT = {
    "INNERCORE": pp.medium.StandardRock,
    "OUTERCORE": pp.medium.StandardRock,
    "MANTLE": pp.medium.StandardRock,
    "ROCK": pp.medium.StandardRock,
    "ICE": pp.medium.Ice,
    "AIR": pp.medium.Air
}

def remove_comments(s: str) -> str:
    if "#" not in s:
        return s
    idx = s.index("#")
    return s[:idx]

def make_particle_definition(particle: Particle) -> pp.particle.ParticleDef:
    '''
    Builds a proposal particle definition

    Parameters
    ----------
    particle: Prometheus particle you want a ParticleDef for

    Returns
    -------
    pdef: PROPOSAL particle definition object corresponing
        to input particle
    '''
    if str(particle) not in 'MuMinus MuPlus EMinus EPlus TauMinus TauPlus'.split():
        raise ValueError(f"Particle string {str(particle)} not recognized")
    pdef = getattr(pp.particle, f'{str(particle)}Def')()
    return pdef


def make_propagator(
    particle: Particle,
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

    pdef = make_particle_definition(particle)
    utilities = make_collection_utilities(
        pdef,
        path_dict["earth file"],
        simulation_specs
    )
    geometries = make_geometries(path_dict["earth file"])
    density_distrs = make_density_distributions(path_dict["earth file"])
    prop = pp.Propagator(pdef, list(zip(geometries, utilities, density_distrs)))

    return prop

def make_geometries(earth_file: str) -> List[pp.Cartesian3D]:
    geometries = []
    with open(earth_file, "r") as f:
        inner_radius = 0
        for line in f:
            if line[0]=="#" or line[0]==" " or line[:1]=="\n":
                continue
            line = remove_comments(line)
            split_line = [x for x in line.replace("\n", "").split(" ") if len(x)>0]
            outer_radius = float(split_line[0])
            geometry = pp.geometry.Sphere(
                pp.Cartesian3D(0,0,1),
                outer_radius * m_to_cm,
                inner_radius * m_to_cm,
            )
            geometries.append(geometry)
            inner_radius = outer_radius

    return geometries
            
def make_density_distributions(earth_file: str) -> List[pp.density_distribution.density_distribution]:
    with open(earth_file, "r") as f:
        inner_radius = 0
        density_distributions = []
        for line in f:
            if line[0]=="#" or line[0]==" " or line[:1]=="\n":
                continue
            line = remove_comments(line)
            split_line = [x for x in line.replace("\n", "").split(" ") if len(x)>0]
            outer_radius = float(split_line[0])
            if len(split_line[4:])==1:
                rho_bar = float(split_line[4])
            else:
                p0 = float(split_line[4])
                p1 = float(split_line[5])
                rho_bar = p0 + p1 * (inner_radius + outer_radius) /2
            density = pp.density_distribution.density_homogeneous(rho_bar)
            density_distributions.append(density)
    return density_distributions


def make_collection_utilities(
    particle_def,
    earth_file,
    simulation_specs
):
    cuts = pp.EnergyCutSettings(
        simulation_specs["ecut"] * GeV_to_MeV,
        simulation_specs["vcut"],
        simulation_specs["continuous randomization"]
    )
    utilities = []
    with open(earth_file, "r") as f:
        inner_radius = 0
        for line in f:
            if line[0]=="#" or line[0]==" " or line[:1]=="\n":
                continue
            line = remove_comments(line)
            split_line = [x for x in line.replace("\n", "").split(" ") if len(x)>0]
            outer_radius = float(split_line[0])
            if len(split_line[4:])==1:
                rho_bar = float(split_line[4])
            else:
                p0 = float(split_line[4])
                p1 = float(split_line[5])
                rho_bar = p0 + p1 * (inner_radius + outer_radius) /2
            test_medium = MEDIUM_DICT[split_line[2]]()
            medium = MEDIUM_DICT[split_line[2]]()
            collection = pp.PropagationUtilityCollection()
            cross = pp.crosssection.make_std_crosssection(
                particle_def,
                medium,
                cuts,
                simulation_specs["interpolate"]
            )
            create_tables = True
            collection.displacement = pp.make_displacement(cross, create_tables)
            collection.interaction = pp.make_interaction(cross, create_tables)
            collection.time = pp.make_time(cross, particle_def, create_tables)
            collection.decay = pp.make_decay(cross, particle_def, create_tables)

            utility = pp.PropagationUtility(collection=collection)
            utilities.append(utility)

    return utilities

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
    else:
        medium_def = getattr(pp.medium, medium_string.capitalize())()
    return medium_def

def init_pp_particle(particle, pdef, coordinate_shift):
    init_state = pp.particle.ParticleState()
    init_state.position = pp.Cartesian3D(
        *(particle.position + coordinate_shift) * m_to_cm
    )
    init_state.energy = particle.e* GeV_to_MeV
    init_state.direction = pp.Cartesian3D(*particle.direction)
    return init_state

def new_proposal_losses(
    prop,
    p_def, 
    particle,
    padding,
    r_inice,
    detector_center,
    coordinate_shift
):
    init_state = init_pp_particle(particle, p_def, coordinate_shift)
    propagation_length = np.linalg.norm(particle.position) + padding
    secondarys = prop.propagate(init_state, propagation_length * m_to_cm)
    continuous_loss_sum  = 0
    for loss in secondarys.stochastic_losses():
        loss_energy = loss.energy * MeV_to_GeV
        if loss.type==1000000008:
            continuous_loss_sum += loss_energy
        else:
            pos = (
                np.array([loss.position.x, loss.position.y, loss.position.z]) * cm_to_m -
                coordinate_shift
            )
            if np.linalg.norm(pos - detector_center) <= r_inice:
                particle.add_loss(Loss(loss.type, loss.energy, pos))
    #continuous_loss_sum = np.sum(secondarys.continuous_losses()) * MeV_to_GeV
    total_dist = secondarys.track_propagated_distances()[-1] * cm_to_m
    # TODO: Add this to config
    cont_resolution = 1.
    loss_dists = np.arange(0, total_dist, cont_resolution)
    # TODO: Remove this really ugly fix
    if len (loss_dists) == 0:
        continuous_loss_sum = 1.* MeV_to_GeV
        total_dist = 1.1
        loss_dists = np.array([0., 1.])
    e_loss = continuous_loss_sum / len(loss_dists)
    for dist in loss_dists:
        pos = dist * particle.direction + particle.position
        particle.add_loss(Loss(1000000008, e_loss, pos))
    for child in secondarys.decay_products():
        particle.add_child(
            particle_from_proposal(child, coordinate_shift, parent=particle)
        )
    return particle

class NewProposalLeptonPropagator(LeptonPropagator):
    """Class for propagating charged leptons with PROPOSAL versions >= 7"""
    def __init__(self, config):
        with open(config["paths"]["earth file"], "r") as f:
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
            self.config["simulation"],
            self.config["paths"]
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
        pdef = make_particle_definition(particle)
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
        propped_particle = new_proposal_losses(
            propagator,
            particle_def,
            particle,
            self._config["simulation"]["propagation padding"],
            detector.outer_radius + 1000.0,
            detector.offset,
            self._coordinate_shift
        )
        return propped_particle
