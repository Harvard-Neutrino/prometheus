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
    if pstr in 'MuMinus MuPlus EMinus EPlus TauMinus TauPlus'.split():
        pdef = getattr(pp.particle, f'{pstr}Def')()
    else:
        raise ValueError(
            f"Particle string {pstr} not recognized"
        )
    return pdef

def _init_dynamic_data(particle, pdef):
    # Function must be called for each new particle.
    # Maybe. Or we might just be able to overwrite the
    # information but that requires some care. Makes me a bit nervous
    particle_dd = pp.particle.DynamicData(pdef.particle_type)
    particle_dd.position = particle.pp_position
    particle_dd.direction = particle.pp_direction
    particle_dd.energy = particle.e * GeV_to_MeV
    return particle_dd

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

def _init_sector(start, end, ecut, vcut):
    #Define a sector
    sec_def = pp.SectorDefinition()
    sec_def.geometry = pp.geometry.Sphere(pp.Vector3D(), end, start)
    sec_def.cut_settings.ecut = ecut
    sec_def.cut_settings.vcut = vcut
    # What should the default behavior of this be ?
    return sec_def

def make_propagator(
    pstr,
    **kwargs
):
    pdef = make_pdef(pstr)
    inner_r = [0, kwargs["r_detector"]]
    outer_r = [kwargs["r_detector"], kwargs["r_max"]]
    ecut = iter_or_rep(kwargs["ecut"])
    vcut = iter_or_rep(kwargs["vcut"])
    medium = _make_medium(kwargs["medium_str"])
    # TODO What is this ?
    detector = pp.geometry.Sphere(
        pp.Vector3D(), kwargs["r_max"] * m_to_cm, 0.0
    )
    sec_defs = []
    for start, end, ecut, vcut in zip(inner_r, outer_r, ecut, vcut):
        start = start * m_to_cm
        end = end * m_to_cm
        sec_def = _init_sector(start, end, ecut, vcut)
        sec_def.medium = medium
        sec_def.particle_location = pp.ParticleLocation.inside_detector
        sec_def.scattering_model = (
            getattr(pp.scattering.ScatteringModel, kwargs["scattering model"])
        )
	    # Bool options
        sec_def.crosssection_defs.brems_def.lpm_effect = kwargs["lpm_effect"]
        sec_def.crosssection_defs.epair_def.lpm_effect = kwargs["lpm_effect"]
        sec_def.do_continuous_randomization = kwargs["continuous_randomization"]
        sec_def.do_continuous_energy_loss_output = kwargs['soft_losses']
        sec_defs.append(sec_def)

    interpolation_def = pp.InterpolationDef()
    # TODO do we need to point these somewhere else ?
    interpolation_def.path_to_tables = (
        "~/.local/share/PROPOSAL/tables"
    )
    interpolation_def.path_to_tables_readonly = (
        "~/.local/share/PROPOSAL/tables"
    )

    prop = pp.Propagator(
        pdef, sec_defs, detector, interpolation_def
    )
    return prop
    

def old_proposal_losses(
    prop,
    pdef,
    particle,
    padding,
    r_inice
):
    particle_dd = _init_dynamic_data(particle, pdef)
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
            if np.linalg.norm(pos) <= r_inice:
                particle.add_loss(Loss(sec.type, sec_energy, pos))
        else: # This is a particle. Might need to propagate
            child = Particle(
                sec.type,
                sec.energy,
                sec.position,
                sec.direction,
                parent=particle
            )
            particle.add_child(child)
    total_loss = None
    return particle
