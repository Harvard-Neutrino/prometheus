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
    "AIR": pp.medium.Air,
    "WATER": pp.medium.Water
}

def remove_comments(s: str) -> str:
    """Helper for removing trailing comments from a string.

    Parameters
    ----------
    s : str
        String you want to remove comments from.

    Returns
    -------
    s: str
        String without the comments.
    """
    if "#" not in s:
        return s
    idx = s.index("#")
    return s[:idx]

def make_particle_definition(particle: Particle) -> pp.particle.ParticleDef:
    """Build a PROPOSAL particle definition.
 
    Parameters
    ----------
    particle : Particle
        Prometheus particle you want a particle definition object for.
 
    Returns
    -------
    pdef : pp.particle.ParticleDef
        PROPOSAL particle definition object corresponding to input particle.
    """
    if str(particle) not in 'MuMinus MuPlus EMinus EPlus TauMinus TauPlus'.split():
        raise ValueError(f"Particle string {particle} not recognized")
    pdef = getattr(pp.particle, f'{particle}Def')()
    return pdef


def make_propagator(
    particle: Particle,
    simulation_specs: dict,
    path_dict: dict
) -> pp.Propagator:
    """Build a PROPOSAL propagator.
 
    Parameters
    ----------
    particle : Particle
        Prometheus particle you want a PROPOSAL propagator for.
    simulation_specs : dict
        Dictionary specifying the configuration settings.
    path_dict : dict
        Dictionary specifying any required path variables.
 
    Returns
    -------
    prop : pp.Propagator
        PROPOSAL propagator for input particle.
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

def make_geometries(earth_file: str) -> List[pp.Cartesian3D]:
    """Build PROPOSAL geometries from an Earth data file.

    Parameters
    ----------
    earth_file : str
        Data file where the parametrization of Earth is stored.

    Returns
    -------
    geometries : list of pp.Cartesian3D
        List of PROPOSAL spherical shells that make up the Earth.
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
            
def make_density_distributions(earth_file: str) -> List[pp.density_distribution.density_distribution]:
    """Create a list of PROPOSAL homogeneous density distributions from an Earth data file.

    Parameters
    ----------
    earth_file : str
        Data file where the parametrization of Earth is stored.

    Returns
    -------
    density_distributions : list of pp.density_distribution.density_distribution
        Density distributions corresponding to the average density in
        each layer of the Earth model at linear order.
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
) -> pp.PropagationUtility:
    """Build PROPOSAL propagation utilities from an Earth file.

    Parameters
    ----------
    particle_def : pp.particle.ParticleDef
        PROPOSAL particle definition.
    earth_file : str
        Data file where the parametrization of Earth is stored.
    simulation_specs : dict
        Dictionary specifying all the simulation specifications.

    Returns
    -------
    utilities : list of pp.PropagationUtility
        List of PROPOSAL ``PropagationUtility`` objects.
    """
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
) -> pp.particle.ParticleState:
    """Initialize a PROPOSAL particle.

    Parameters
    ----------
    particle : Particle
        Prometheus particle you want to create the PROPOSAL state for.
    coordinate_shift : np.ndarray
        Difference between the PROPOSAL coordinate system centered on
        the Earth's center and the Prometheus coordinate system in
        meters.

    Returns
    -------
    init_state : pp.particle.ParticleState
        PROPOSAL particle state with energy, position, and direction
        matching the input particle.
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
    """Propagate a Prometheus particle using PROPOSAL.

    This modifies the particle losses in place.

    Parameters
    ----------
    prop : pp.Propagator
        PROPOSAL propagator corresponding to the input particle.
    particle : Particle
        Prometheus particle to propagate.
    padding : float
        Propagation padding in meters. The propagation distance is
        calculated as
        ``numpy.linalg.norm(particle.position - detector_center) + padding``.
    r_inice : float
        Distance from the center of the edge detector where losses should be recorded.
    detector_center : np.ndarray
        Center of the detector in the Prometheus coordinate system in
        meters.
    coordinate_shift : np.ndarray
        Difference between the PROPOSAL coordinate system centered on
        the Earth's center and the Prometheus coordinate system in
        meters. The norm of this vector should be the radius
        between the center of the Earth and the start of the atmosphere,
        and should usually only have a z-component.
    """
    init_state = init_pp_particle(particle, coordinate_shift)
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
            # TODO more this to the serialization function. DTaSD
            if np.linalg.norm(pos - detector_center) <= r_inice:
                particle.losses.append(
                    Loss(loss.type, loss_energy, pos)
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
        particle.losses.append(Loss(1000000008, e_loss, pos))
    for child in secondarys.decay_products():
        particle.children.append(
            particle_from_proposal(child, coordinate_shift, parent=particle)
        )

class NewProposalLeptonPropagator(LeptonPropagator):
    """Propagate charged leptons with PROPOSAL versions >= 7."""
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
        """Create a PROPOSAL propagator for a Prometheus particle."""
        propagator = make_propagator(
            particle,
            self.config["simulation"],
            self.config["paths"]
        )
        return propagator

    def _make_particle_def(self, particle: Particle):
        """Create a PROPOSAL particle definition for a Prometheus particle."""
        pdef = make_particle_definition(particle)
        return pdef

    def energy_losses(
        self, 
        particle: Particle,
        detector: Detector
    ) -> None:
        """Propagate a particle and track the losses.

        Losses and children are added in place.

        Parameters
        ----------
        particle : Particle
            Prometheus particle that should be propagated.
        detector : Detector
            Detector that this is being propagated within.
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
