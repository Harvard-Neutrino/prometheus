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

import warnings
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

## !! uncomment following block if you wish to see logger line (TR f2k print line) in this file: !!

# if not logger.handlers:
#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('%(levelname)s: %(message)s')
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)



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
        raise ValueError(f"Particle string {particle} not recognized")
    pdef = getattr(pp.particle, f'{particle}Def')()
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

    pp.InterpolationSettings.tables_path = path_dict["tables path"]
    pdef = make_particle_definition(particle)
    utilities = make_propagation_utilities(
        pdef,
        path_dict["earth model location"],
        simulation_specs
    )
    geometries = make_geometries(path_dict["earth model location"])
    density_distrs = make_density_distributions(path_dict["earth model location"])
    prop = pp.Propagator(pdef, list(zip(geometries, utilities, density_distrs)))

    return prop

# Pydoc crashes if you remove the comment :-(
def make_geometries(earth_file: str) -> List:
#def make_geometries(earth_file: str) -> List[pp.Cartesian3D]:
    """Make list of proposal geometries from earth datafile

    params
    ______
    earth_file: data file where the parametrization of Earth is stored
    
    returns
    _______
    geometries: List of PROPOSAL spherical shells that make up the Earth
    """
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
            
# Once again, uncomment messes with pydoc :-(
def make_density_distributions(earth_file: str) -> List:
#def make_density_distributions(earth_file: str) -> List[pp.density_distribution.density_distribution]:
    """Make list of proposal homogeneous density distributions from
    Earth datafile

    params
    ______
    earth_file: data file where the parametrization of Earth is stored

    returns
    _______
    density_distributions: Density distributions corresponding to the 
        average density in each layer of the Earth model at linear order
    """
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


def make_propagation_utilities(
    particle_def: pp.particle.ParticleDef,
    earth_file: str,
    simulation_specs: dict
):
# Pydoc again. Once day we will triumph
#) -> pp.PropagationUtility:
    """Make a list of PROPOSAL propagation utilities from an earth file
        for a particle given some simulation specifications

    params
    ______
    particle_def: PROPOSAL particle definition
    earth_file: data file where the parametrization of Earth is stored
    simulation_specs: dictionary specifying all the simulation specifications

    returns
    _______
    utilities: List of PROPOSAL PropagationUtility objects
    """
    ### ppc 'amu' parameterisations are from 0.5 GeV Ecut GEANT4 simulations, so anything other than Ecut=0.5 GeV would mismodel light yield from muons
    ## should find a better way to deal with these settings (perhaps even take away user option to change them), but for now:
    if simulation_specs["ecut"] != 0.5 or simulation_specs["vcut"] != 1:
        warnings.warn("Adjusting ecut and vcut to work with ppc", UserWarning)
        simulation_specs["ecut"] = 0.5
        simulation_specs["vcut"] = 1
         

    cuts = pp.EnergyCutSettings(
        simulation_specs["ecut"] * GeV_to_MeV,
        simulation_specs["vcut"],
        simulation_specs["continuous randomization"]
    )
    utilities = []
    with open(earth_file, "r") as f:
        # inner_radius = 0
        for line in f:
            if line[0]=="#" or line[0]==" " or line[:1]=="\n":
                continue
            line = remove_comments(line)
            split_line = [x for x in line.replace("\n", "").split(" ") if len(x)>0]
            # outer_radius = float(split_line[0])
            if len(split_line[4:])==1:
                # TODO: Get the feeling these should be used but aren't
                gibberish = None
                # rho_bar = float(split_line[4])
            else:
                p0 = float(split_line[4])
                p1 = float(split_line[5])
                # TODO: Get the feeling these should be used but aren't
                # rho_bar = p0 + p1 * (inner_radius + outer_radius) /2
            # test_medium = MEDIUM_DICT[split_line[2]]()
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
            if simulation_specs["decay"]:
                collection.decay = pp.make_decay(cross, particle_def, create_tables)

            utility = pp.PropagationUtility(collection=collection)
            utilities.append(utility)

    return utilities

def init_pp_particle(
    particle: Particle,
    #pdef: pp.particle.ParticleDef,
    coordinate_shift: np.ndarray
):
# Pydoc
#) -> pp.particle.ParticleState:
    """Initialize a PROPOSAL particle

    params
    ______
    particle: Prometheus particle for which to make the PROPOSAL state
    pdef: PROPOSAL particle definition
    coordinate_shift: Difference between the PROPOSAL coordinate system
        centered on the the Earth's center and Prometheus coordinate
        system in meters. The norm of this vector should be the radius
        between the center of the Earth and the start of the atmoshphere

    returns
    _______
    init_state: PROPOSAL particle state with energy, position, and direction
        matching input particle
    """
    init_state = pp.particle.ParticleState()
    init_state.position = pp.Cartesian3D(
        *(particle.position + coordinate_shift) * m_to_cm
    )
    init_state.energy = particle.e * GeV_to_MeV
    init_state.direction = pp.Cartesian3D(*particle.direction)
    return init_state

# TODO Sorry about this function:-(
def new_proposal_losses(
    prop: pp.Propagator,
    particle: Particle,
    padding: float,
    r_inice: float,
    detector_center: np.ndarray,
    coordinate_shift: np.ndarray
) -> None:
    """Propagate a Prometheus particle using PROPOSAL, modifying the particle
    losses in place

    params
    ______
    prop: Proposal propagator porresponding to the input particle
    particle: Prometheus particle to propagate
    padding: propagation padding in meters. The propagation distance is calcuated as:
        np.linalg.norm(particle.position - detector_center) + padding
    detector_center: Center of the detector in Prometheus coordinate system in meters
    coordinate_shift: Difference between the PROPOSAL coordinate system
        centered on the the Earth's center and Prometheus coordinate
        system in meters. The norm of this vector should be the radius
        between the center of the Earth and the start of the atmoshphere, and 
        should usually only have a z-component
    """
    init_state = init_pp_particle(particle, coordinate_shift)
    propagation_length = np.linalg.norm(particle.position) + padding
    secondarys = prop.propagate(init_state, propagation_length * m_to_cm)

    if particle.pdg_code == 13: ## TODO should find a way to remove this if statement and combine with non-amu loss types
        for i in range(1, len(secondarys.track_types())):
            time = secondarys.track_times()[i]
            l=0
            l = (secondarys.track_propagated_distances()[i] - secondarys.track_propagated_distances()[i-1])*cm_to_m # to m 
            x = secondarys.track_positions()[i-1][0]
            y = secondarys.track_positions()[i-1][1]
            z = secondarys.track_positions()[i-1][2]
            x_dir = secondarys.track_directions()[i-1][0] # prometheus not currently using (for logging purposes)
            y_dir = secondarys.track_directions()[i-1][1] # prometheus not currently using (for logging purposes)
            z_dir = secondarys.track_directions()[i-1][2] # prometheus not currently using (for logging purposes)
            time = secondarys.track_times()[i-1] # prometheus not currently using (for logging purposes)
            pos = (
                np.array([x, y, z]) * cm_to_m -
                coordinate_shift
            )
            energy = (secondarys.track_energies()[i-1] - secondarys.track_energies()[i])*MeV_to_GeV # to gev
            theta = np.arctan2(np.sqrt(x_dir*x_dir + y_dir*y_dir), z_dir) # prometheus not currently using (for logging purposes)
            phi = np.arctan2(y_dir, x_dir) # prometheus not currently using (for logging purposes)
            loss_type = secondarys.track_types()[i].value

            
            ## quick fix for now. Reason for change of type is: 1000000008 is also used for electron and tau continous losses, and I'm not sure what those losses should be, but probably not amu... 
            if loss_type == 1000000008: 
                loss_type=1000000018 ## 1000000018 is defined in translators as 'amu-'
                energy = secondarys.track_energies()[i-1]*MeV_to_GeV # in case of muon track, energy should be energy of muon at start of track

            ### decay type (child decay particles handled later)
            if loss_type != 1000000011: 
                logger.debug(f"New proposal f2k line: TR {0} {0} {loss_type} {x*cm_to_m} {y*cm_to_m} {z*cm_to_m} {theta} {phi} {l} {energy} {time}\n") ## just for logging purposes (matches write_to_f2k.py TR line except z which is expected)
                particle.losses.append(Loss(loss_type, energy, pos, l)) # this time using track length

    else: ## not changing loss code for any particles other than muon (for now)
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
                # TODO more this to the serialization function. DTaSD
                if np.linalg.norm(pos - detector_center) <= r_inice:
                    particle.losses.append(
                        Loss(loss.type, loss_energy, pos, 0) ## 0m  track length
                    )
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
            particle.losses.append(Loss(1000000008, e_loss, pos, 0)) ## 0 m track length
    for child in secondarys.decay_products():
        particle.children.append(
            particle_from_proposal(child, coordinate_shift, parent=particle)
        )

class NewProposalLeptonPropagator(LeptonPropagator):
    """Class for propagating charged leptons with PROPOSAL versions >= 7"""
    def __init__(self, config):
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
    ) -> None:
        """Propagate a particle and track the losses. Losses and children
        are added in place

        params
        ______
        particle: Prometheus Particle that should be propagated
        detector: Detector that this is being propagated within

        returns
        _______
        propped_particle: Prometheus Particle after propagation
        """
        # TODO particle_def is not needed. DTaSD
        particle_def, propagator = self[particle]
        propped_particle = new_proposal_losses(
            propagator,
            particle,
            self._config["simulation"]["propagation padding"],
            detector.outer_radius + 1000.0,
            detector.offset,
            self._coordinate_shift
        )
