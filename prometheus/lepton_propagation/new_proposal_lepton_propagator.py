# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,

import numpy as np
import proposal as pp

from .lepton_propagator import LeptonPropagator
from ..particle import Particle
from ..detector import Detector
from ..utils.units import GeV_to_MeV, MeV_to_GeV, cm_to_m, m_to_cm

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
    # Make the PROPOSAL utilities object
    ecut = simulation_specs["ecut"]
    vcut = simulation_specs["vcut"]
    if hasattr(vcut, "__iter__") and len(ecut)!=2:
        raise ValueError("You can only specify vcut lists of length 2")
    if hasattr(ecut, "__iter__") and len(ecut)!=2:
        raise ValueError("You can only specify ecut lists of length 2")

    pdef = make_particle_definition(particle)
    medium = make_medium(simulation_specs["medium"])
    # TODO figure out how to get the rest of the options working
    do_continuous_randomization = simulation_specs["continuous randomization"]
    # List have been passed for both energy cut settings
    # TODO: Check what the hell this is
    interpolate = True
    if hasattr(vcut, "__iter__") and hasattr(ecut, "__iter__"):
        utilities = [
            make_utility(
                pdef,
                interpolate,
                pp.EnergyCutSettings(
                    ec * GeV_to_MeV,
                    vc,
                    do_continuous_randomization),
                medium
            )
        for ec, vc in zip(ecut, vcut)]
    # A List was passed for vcut only
    elif hasattr(vcut, "__iter__"):
        utilities = [
            make_utility(
                pdef,
                interpolate,
                pp.EnergyCutSettings(
                    ecut * GeV_to_MeV,
                    vc,
                    do_continuous_randomization
                ),
                medium
            )
        for vc in vcut]
    # A List was passed for ecut only
    elif hasattr(ecut, "__iter__"):
        utilities = [
            make_utility(
                pdef,
                interpolate,
                pp.EnergyCutSettings(
                    ec * GeV_to_MeV,
                    vcut,
                    do_continuous_randomization
                ),
                medium
            )
        for ec in ecut]
    # Scalars were passed for both energy cut settings
    else:
        utilities = [
            make_utility(
                pdef,
                interpolate,
                pp.EnergyCutSettings(
                    ecut * GeV_to_MeV,
                    vcut,
                    do_continuous_randomization
                ),
                medium
            )
        for _ in range(2)]

    # Make the PROPOSAL geometries
    inner_r = [0, simulation_specs["inner radius"]]
    outer_r = [simulation_specs["inner radius"], simulation_specs["maximum radius"]]
    geometries = [
        pp.geometry.Sphere(
            pp.Cartesian3D(),
            end * m_to_cm,
            start * m_to_cm
        )
        for start, end in zip(outer_r, inner_r)
    ]

    # Make the density distributions
    density_distrs = [
        pp.density_distribution.density_homogeneous(
            medium.mass_density
        ) for _ in range(2)
    ]

    prop = pp.Propagator(pdef, list(zip(geometries, utilities, density_distrs)))

    return prop

def make_utility(
    particle_def,
    interpolate,
    cuts,
    medium
):
    collection = pp.PropagationUtilityCollection()
    cross = pp.crosssection.make_std_crosssection(
        particle_def,
        medium,
        cuts,
        interpolate
    )
    # TODO look into continuous randomization
    create_tables = True
    collection.displacement = pp.make_displacement(cross, create_tables)
    collection.interaction = pp.make_interaction(cross, create_tables)
    collection.time = pp.make_time(cross, particle_def, create_tables)
    collection.decay = pp.make_decay(cross, particle_def, create_tables)

    utility = pp.PropagationUtility(collection=collection)

    return utility

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

def init_pp_particle(particle, pdef):
    # Function must be called for each new particle.
    # Maybe. Or we might just be able to overwrite the
    # information but that requires some care. Makes me a bit nervous
    init_state = pp.particle.ParticleState()
    init_state.position = pp.Cartesian3D(
        *particle.position
    )
    init_state.energy = particle.energy * GeV_to_MeV
    init_state.direction = pp.Cartesian3D(*particle.direction)
    return init_state

def new_proposal_losses(
    prop,
    p_def, 
    particle,
    padding,
    r_inice,
    detector_center
):
    # TODO: Check if init state needs to be set
    init_state = init_pp_particle(particle, p_def)
    propagation_length = np.linalg.norm(particle.position) + padding
    secondarys = prop.propagate(init_state, propagation_length * m_to_cm)
    continuous_loss  = 0
    for loss in secondarys.stochastic_losses():
        loss_energy = loss.energy * MeV_to_GeV
        if loss.types==1000000008:
            continuous_loss += loss_energy
        else:
            if np.linalg.norm(pos - detector_center) <= r_inice:
                pos = np.array([loss.position.x, loss.position.y, loss.position.z]) * cm_to_m
                particle.add_loss(Loss(sec.type, sec_energy, pos))
    # TODO: Update this ugly fix
    cont_loss_sum = np.sum(secondarys.continuous_losses()) * MeV_to_GeV
    total_dist = secondarys.track_propagated_distances()[-1] * cm_to_m
    # TODO: Add this to config
    cont_resolution = 1.
    loss_dists = np.arange(0, total_dist, cont_resolution)
    # TODO: Remove this really ugly fix
    if len (loss_dists) == 0:
        cont_loss_sum = 1.* MeV_to_GeV
        total_dist = 1.1
        loss_dists = np.array([0., 1.])
    e_loss = cont_loss_sum / len(loss_dists)
    for dist in loss_dists:
        pos = dist * particle.direction + particle.position
        particle.add_loss(Loss(1000000008, e_loss, pos))
    if soft_losses:
        total_loss = np.sum([len(losses[loss]) for loss in losses])
    else:
        total_loss = np.sum([len(losses[loss]) for loss in losses if loss != 'continuous'])
    return particle

class NewProposalLeptonPropagator(LeptonPropagator):
    """Class for propagating charged leptons with PROPOSAL versions >= 7"""
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
            detector.offset
        )
        return propped_particle
