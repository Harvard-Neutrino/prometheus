# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the different lepton propagators

import numpy as np
from hebe.config import config
from packaging.version import parse
from hebe.particle import Particle

try:
    import proposal as pp
except ImportError:
    raise ImportError('Could not import proposal!')

def energy_losses(
    prop,
    pdef,
    particle,
    padding
):
    '''
    p_energies: particle energy array in units of GeV
    mcp_def: PROPOSAL mCP definition
    direction: (vx,vy,vz)
    propagation_length: cm
    prop : use a pregenerated propagator object if provided.
    Else will make one on the fly.
    '''
    if int(pp.__version__.split('.')[0]) < 7:
        from .old_proposal import old_proposal_losses
        return old_proposal_losses(
            prop,
            pdef,
            particle,
            padding
        )
    else:
        from .new_proposal import new_proposal_losses
        return _new_proposal_losses(
            prop,
            p_def,
            energy,
            soft_losses,
            direction=direction,
            position=position,
            propagation_length=propagation_length
        )


class LP(object):
    ''' Interface class to the different lepton propagators
    '''
    def __init__(self):
        if config['lepton propagator']['name'] == 'proposal':
            self.args = {}
            print('Using proposal')
            self.__pp_version = pp.__version__
            print('The proposal version is ' + str(self.__pp_version))
            self.kwargs = config["lepton propagator"].copy()
            self.kwargs["r_detector"] = config["detector"]["radius"]
            self.kwargs["r_max"] = config["detector"]["r_max"]
            self.kwargs["medium_str"] = config["detector"]["medium"]
            if parse(self.__pp_version) > parse('7.0.0'):
                print('Using new setup')
                self.prop_dict = self._new_proposal_setup()
            else:
                print('Using a version before 7.0.0')
                self.prop_dict = self._make_old_propdict(
                    config["lepton propagator"]["lepton"],
                    **self.kwargs
                )
                self.pdef_dict = self._make_old_pdefdict(
                    config["lepton propagator"]["lepton"]
                )
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
                self.args['ecut'], self.args['vcut'],
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

    # TODO make this two functions ?
    def _make_old_propdict(self, pstr, **kwargs):
        """Set up a proposal propagator for version <= 6"""
        print('----------------------------------------------------')
        from .old_proposal import make_propagator
        if pstr in "EMinus EPlus MuMinus MuPlus".split():
            prop_dict = {pstr: make_propagator(pstr, **kwargs)}
        else: # Taus can produce other leptons
            if "Tau" not in pstr:
                raise Exception("What is happening here ?????")
            prop_dict = {pstr: make_propagator(pstr, **kwargs)}
            pstr = "E" + pstr[3:]
            prop_dict[pstr] = make_propagator(pstr, **kwargs)
            pstr = "Mu" + pstr[1:]
            prop_dict[pstr] = make_propagator(pstr, **kwargs)

        return prop_dict

    def _make_old_pdefdict(self, pstr):
        """Set up a proposal propagator for version <= 6"""
        print('----------------------------------------------------')
        from .old_proposal import make_pdef
        if pstr in "EMinus EPlus MuMinus MuPlus".split():
            pdef_dict = {pstr: make_pdef(pstr)}
        else: # Taus can produce other leptons
            if "Tau" not in pstr:
                raise Exception("What is happening here ?????")
            pdef_dict = {pstr: make_pdef(pstr)}
            pstr = "E" + pstr[3:]
            pdef_dict[pstr] = make_pdef(pstr)
            pstr = "Mu" + pstr[1:]
            pdef_dict[pstr] = make_pdef(pstr)

        return pdef_dict

    def energy_losses(self, particle):
        losses = energy_losses(
            self.prop_dict,
            self.pdef_dict,
            particle,
            self.kwargs["propagation padding"]
        )
        return losses
