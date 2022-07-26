# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the different lepton propagators

import numpy as np
import proposal as pp

from ..config import config
from .loss import Loss
from ..particle import Particle
from ..utils import iter_or_rep
from  ..utils.units import GeV_to_MeV, cm_to_m, m_to_cm, MeV_to_GeV

def make_pdef(pstr):
    '''
    Builds a proposal particle definition

    Parameters
    ----------
    pstr: string
        String which defines the medium in which proposal should propagate

    Returns
    -------
    pdef : proposal.particle.ParticleDef
        PROPOSAL particle definition object corresponing to pstr
    '''
    if pstr in 'MuMinus MuPlus EMinus EPlus TauMinus TauPlus'.split():
        pdef = getattr(pp.particle, f'{pstr}Def')()
    else:
        raise ValueError(
            f"Particle string {pstr} not recognized"
        )
    return pdef

def make_propagator(
    pstr,
    **kwargs
):
    # Make the PROPOSAL utilities object
    ecut = kwargs["ecut"]
    vcut = kwargs["vcut"]
    if hasattr(vcut, "__iter__") and len(ecut)!=2:
        raise ValueError("You can only specify vcut lists of length 2")
    if hasattr(ecut, "__iter__") and len(ecut)!=2:
        raise ValueError("You can only specify ecut lists of length 2")

    pdef = make_pdef(pstr)
    medium = _make_medium(kwargs["medium_str"])
    # TODO figure out how to get the rest of the options working
    do_continuous_randomization = kwargs["continuous_randomization"]
    # List have been passed for both energy cut settings
    if hasattr(vcut, "__iter__") and hasattr(ecut, "__iter__"):
        utilities = [
            make_utility(
                pdef,
                interpolate,
                pp.EnergyCutSettings(ec, vc, do_continuous_randomization),
                medium
            )
        for ec, vc in zip(ecut, vcut)]
    # A List was passed for vcut only
    elif hasattr(vcut, "__iter"):
        utilities = [
            make_utility(
                pdef,
                interpolate,
                pp.EnergyCutSettings(ecut, vc, do_continuous_randomization),
                medium
            )
        for vc in vcut]
    # A List was passed for ecut only
    elif hasattr(ecut, "__iter"):
        utilities = [
            make_utility(
                pdef,
                interpolate,
                pp.EnergyCutSettings(ec, vcut, do_continuous_randomization),
                medium
            )
        for ec in ecut]
    # Scalars were passed for both energy cut settings
    else:
        utilities = [
            make_utility(
                pdef,
                interpolate,
                pp.EnergyCutSettings(ecut, vcut, do_continuous_randomization),
                medium
            )
        for _ in range(2)]

    # Make the PROPOSAL geometries
    inner_r = [0, kwargs["r_detector"]]
    outer_r = [kwargs["r_detector"], kwargs["r_max"]]
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

    prop = pp.Propagator(
        particle_def,
        list(zip(geometries, utilities, density_distrs))
    )

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

def _make_medium(medium_string):
    '''
    Builds a proposal medium

    Parameters
    ----------
    medium_string: string
        String which defines the medium in which proposal should propagate

    Returns
    -------
    medium_def : PROPOSAL medium object
        Medium in which the propagation should take place
    '''
    print('This assumes a homogeneous medium!')
    if medium_string.lower() not in 'water ice'.split():
        raise ValueError(f"Medium {medium_string} not supported at this time.")
    else:
        medium_def = getattr(pp.medium, medium_string.capitalize())()
    return medium_def

def _init_pp_particle(particle, pdef):
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

def _new_proposal_losses(
    prop,
    p_def, 
    particle,
    padding,
    r_inice
):
    # TODO: Check if init state needs to be set
    init_state = _init_pp_particle(particle, pdef)
    propagation_length = np.linalg.norm(particle.position) + padding
    secondarys = prop.propagate(init_state, propagation_length * m_to_cm)

    for loss in secondarys.stochastic_losses():
        loss_energy = loss.energy * MeV_to_GeV
        pos = np.array([loss.position.x, loss.position.y, loss.position.z]) * cm_to_m
        if loss_type_name == pp.particle.Interaction_Type.epair:
            losses['epair'].append([log_sec_energy, pos])
        elif loss_type_name == pp.particle.Interaction_Type.brems:
            losses['brems'].append([log_sec_energy, pos])
        elif loss_type_name == pp.particle.Interaction_Type.ioniz:
            losses['ioniz'].append([log_sec_energy, pos])
        elif loss_type_name == pp.particle.Interaction_Type.photonuclear:
            losses['photo'].append([log_sec_energy, pos])
        else:
            pass

    # TODO: Update this ugly fix
    cont_loss_sum = np.sum(secondarys.continuous_losses()) * MeV_to_GeV
    total_dist = secondarys.track_propagated_distances()[-1] * cm_to_m
    # TODO: Add this to config
    cont_resolution = 1.
    loss_dists = np.arange(0, total_dist, cont_resolution)
    # TODO: Remove this really ugly fix
    if len (loss_dists) == 0:
        cont_loss_sum = 1. * GeV_to_MeV
        total_dist = 1.1
        loss_dists = np.array([0., 1.])
    e_loss = cont_loss_sum / len(loss_dists)
    losses['continuous'] = ([
        [e_loss,
         dist * particle.direction + particle.position]
        for dist in loss_dists])
    # TODO This should probably be an awkward array
    losses['continuous'] = np.array(losses['continuous'], dtype='object')
    losses['brems'] = np.array(losses['brems'], dtype='object')
    losses['epair'] = np.array(losses['epair'], dtype='object')
    losses['photo'] = np.array(losses['photo'], dtype='object')
    losses['ioniz'] = np.array(losses['ioniz'], dtype='object')
    if soft_losses:
        total_loss = np.sum([len(losses[loss]) for loss in losses])
    else:
        total_loss = np.sum([len(losses[loss]) for loss in losses if loss != 'continuous'])
    return losses, total_loss

