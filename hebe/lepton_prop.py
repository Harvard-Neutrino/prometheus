# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the different lepton propagators

import numpy as np
from .config import config
from packaging.version import parse
try:
    import proposal as pp
except ImportError:
    raise ImportError('Could not import proposal!')


def jeff_losses(
    prop,
    p_def, 
    energy, 
    soft_losses,
    direction=(0, 0, -1), 
    position=(0,0,0),
    propagation_length=1e5
):
    '''
    p_energies: particle energy array in units of GeV
    mcp_def: PROPOSAL mCP definition
    direction: (vx,vy,vz)
    propagation_length: cm
    prop : use a pregenerated propagator object if provided.
    Else will make one on the fly.
    '''
    
    losses = {}
    losses['continuous'] = []
    losses['epair'] = []
    losses['brems'] = []
    losses['ioniz'] = []
    losses['photo'] = []
    # TODO move this to units file or some shit
    GeV_to_MeV = 1e3
    MeV_to_GeV = 1e-3
    m_to_cm = 1e2
    cm_to_m = 1/m_to_cm

    position = position * m_to_cm
    propagation_length = propagation_length * m_to_cm
    # particle = pp.particle.DynamicData(p_def.particle_type)
    # particle.position = pp.Vector3D(*position)
    # particle.direction = pp.Vector3D(*direction)
    # particle.propagated_distance = 0
    # particle.energy = energy * GeV_to_MeV
    # secondarys = prop.propagate(particle, propagation_length)
    # TODO: Check if init state needs to be set
    init_state = pp.particle.ParticleState()
    init_state.position = pp.Cartesian3D(
        *position
    )
    init_state.energy = energy * GeV_to_MeV
    init_state.direction = pp.Cartesian3D(*direction)
    secondarys = prop.propagate(init_state, propagation_length)  # cm

    # for sec in secondarys.particles:
    #     log_sec_energy = np.log10((sec.parent_particle_energy - sec.energy) * MeV_to_GeV)
    #     pos = np.array([sec.position.x, sec.position.y, sec.position.z])*cm_to_m
    #     # TODO make this more compact with dict
    #     if sec.type == int(pp.particle.Interaction_Type.ContinuousEnergyLoss):
    #         losses['continuous'].append([log_sec_energy, pos])
    #     elif sec.type == int(pp.particle.Interaction_Type.Epair):
    #         losses['epair'].append([log_sec_energy, pos])
    #     elif sec.type == int(pp.particle.Interaction_Type.Brems):
    #         losses['brems'].append([log_sec_energy, pos])
    #     elif sec.type == int(pp.particle.Interaction_Type.DeltaE):
    #         losses['ioniz'].append([log_sec_energy, pos])
    #     elif sec.type == int(pp.particle.Interaction_Type.NuclInt):
    #         losses['photo'].append([log_sec_energy, pos])
    #     else:
    #         pass
    # New interface
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

def _make_particle(p_string: str):
    ''' Builds a proposal particle

    Parameters
    ----------
    p_string : str
        Name of the particle to generate in the LeptonInjector convention

    Returns
    -------
    p_def : pp.particle
        A proposal particle
    '''
    if p_string in 'MuMinus MuPlus EMinus EPlus TauMinus TauPlus'.split():
        p_def = getattr(pp.particle, f'{p_string}Def')()
    else:
        raise ValueError(
            f"Particle string {particle_string} not recognized"
        )
    return p_def


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


class LP(object):
    ''' Interface class to the different lepton propagators
    '''
    def __init__(self):
        if config['lepton propagator']['name'] == 'proposal':
            self.args = {}
            print('Using proposal')
            self.__pp_version = pp.__version__
            print('The proposal version is ' + str(self.__pp_version))
            # TODO Make so that lepton is tied to the LI situation / set by it
            self.args['p_def'] = _make_particle(config['lepton propagator']['lepton'])
            self.args['target'] = _make_medium(config['lepton propagator']['medium'])
            self.args['e_cut'] = config['lepton propagator']['e_cut']
            self.args['v_cut'] = config['lepton propagator']['v_cut']
            self.args['soft_losses'] = (
                config["lepton propagator"]['soft_losses']
            )
            if parse(self.__pp_version) > parse('7.0.0'):
                print('Using new setup')
                self.prop = self._new_proposal_setup()
            else:
                print('Using a version before 7.0.0')
                self.prop = self._old_proposal_setup()
        else:
            raise ValueError(
                'Unknown lepton propagator! Check the config file'
            )

    def _new_proposal_setup(self):
        """Set up a proposal propagator for version > 7"""
        print('----------------------------------------------------')
        print('Setting up proposal')
        args = {
            "particle_def": self.args['p_def'],
            "target": self.args['target'],
            "interpolate": config['lepton propagator']['interpolation'],
            "cuts": pp.EnergyCutSettings(
                self.args['e_cut'], self.args['v_cut'],
                self.args['soft_losses']),
        }
        cross = pp.crosssection.make_std_crosssection(
            **args
        )  # use the standard crosssections
        collection = pp.PropagationUtilityCollection()

        collection.displacement = pp.make_displacement(cross, True)
        collection.interaction = pp.make_interaction(cross, True)
        collection.time = pp.make_time(cross, args["particle_def"], True)

        utility = pp.PropagationUtility(collection=collection)

        detector = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1e20)
        density_distr = pp.density_distribution.density_homogeneous(
            args["target"].mass_density
        )
        prop = pp.Propagator(
            args["particle_def"], [(detector, utility, density_distr)]
        )
        print('Finished setup')
        print('----------------------------------------------------')
        return prop

    def _old_proposal_setup(self):
        """Set up a proposal propagator for version <= 6"""
        print('----------------------------------------------------')
        detector = pp.geometry.Sphere(pp.Vector3D(), 1.0e20, 0.0)
        sector_def = pp.SectorDefinition()
        sector_def.cut_settings = pp.EnergyCutSettings(
            self.args['e_cut'],
            self.args['v_cut']
        )
        sector_def.medium = self.args['target']
        sector_def.geometry = detector
        sector_def.scattering_model = (
            pp.scattering.ScatteringModel.NoScattering
        )
        sector_def.crosssection_defs.brems_def.lpm_effect = False
        sector_def.crosssection_defs.epair_def.lpm_effect = False
        sector_def.do_continuous_energy_loss_output = self.args['soft_losses']

        interpolation_def = pp.InterpolationDef()
        # TODO do we need to point these somewhere else ?
        interpolation_def.path_to_tables = (
            "~/.local/share/PROPOSAL/tables"
        )
        interpolation_def.path_to_tables_readonly = (
            "~/.local/share/PROPOSAL/tables"
        )

        prop = pp.Propagator(
            self.args['p_def'], [sector_def], detector, interpolation_def
        )
        return prop

    def j_losses(self, event, prop_len):
        losses = jeff_losses(
            self.prop,
            self.args['p_def'], 
            event['energy'],
            self.args['soft_losses'],
            direction=event['dir'], 
            position=event['pos'],
            propagation_length=prop_len
        )
        return losses
