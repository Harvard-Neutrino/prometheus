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

def _new_proposal_losses(
    prop,
    p_def, 
    energy, 
    soft_losses,
    direction=(0, 0, -1), 
    position=(0,0,0),
    propagation_length=1e5
):
    losses = {}
    losses['continuous'] = []
    losses['epair'] = []
    losses['brems'] = []
    losses['ioniz'] = []
    losses['photo'] = []
    position = position * m_to_cm
    propagation_length = propagation_length * m_to_cm
    # TODO: Check if init state needs to be set
    init_state = pp.particle.ParticleState()
    init_state.position = pp.Cartesian3D(
        *position
    )
    init_state.energy = energy * GeV_to_MeV
    init_state.direction = pp.Cartesian3D(*direction)
    secondarys = prop.propagate(init_state, propagation_length)  # cm

    for loss in secondarys.stochastic_losses():
        # dist = loss.position.z / 100
        e_loss = loss.energy / 1e3

        """
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        
        p = position + dist * direction
        t = dist / Constants.c_vac + time
        """
        # dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        log_sec_energy = np.log10(loss.energy * MeV_to_GeV)
        pos = np.array([loss.position.x, loss.position.y, loss.position.z]) * cm_to_m
        # TODO make this more compact with dict
        loss_type_name = pp.particle.Interaction_Type(loss.type)
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
    cont_loss_sum = sum([np.log10(loss.energy * MeV_to_GeV) for loss in secondarys.continuous_losses()])
    total_dist = secondarys.track_propagated_distances()[-1] * cm_to_m
    # TODO: Add this to config
    cont_resolution = 1.
    loss_dists = np.arange(0, total_dist, cont_resolution)
    # TODO: Remove this really ugly fix
    if len (loss_dists) == 0:
        cont_loss_sum = 1.
        total_dist = 1.1
        loss_dists = np.array([0., 1.])
    e_loss = cont_loss_sum / len(loss_dists)
    # TODO: Check direction and position
    losses['continuous'] = ([
        [e_loss,
         dist * direction + position]
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

