# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the different lepton propagators

import numpy as np
from ..config import config
from packaging.version import parse

try:
    import proposal as pp
except ImportError:
    raise ImportError('Could not import proposal!')

def energy_losses(
    prop,
    pdef,
    particle,
    padding,
    r_inice,
    detector_center
):
    '''
    '''
    if int(pp.__version__.split('.')[0]) < 7:
        from .old_proposal import old_proposal_losses
        return old_proposal_losses(
            prop,
            pdef,
            particle,
            padding,
            r_inice,
            detector_center
        )
    else:
        from .new_proposal import _new_proposal_losses
        return _new_proposal_losses(
            prop,
            pdef,
            particle,
            padding,
            r_inice,
            detector_center
        )


class LP(object):
    ''' Interface class to the different lepton propagators
    '''
    def __init__(self):
        if 'proposal' in config['lepton propagator']['name']:
            self._prop_dict = {}
            self._pdef_dict = {}
            print('Using proposal')
            self._kwargs = config["lepton propagator"].copy()
            self._kwargs["r_detector"] = config["detector"]["radius"]
            self._kwargs["r_max"] = config["detector"]["r_max"]
            self._kwargs["medium_str"] = config["detector"]["medium"]
            self.__pp_version = pp.__version__
            print('The proposal version is ' + str(self.__pp_version))
            if parse(self.__pp_version) > parse('7.0.0'):
                print('Using new setup')
                from .new_proposal import make_propagator, make_pdef
                self._make_prop = make_propagator
                self._make_pdef = make_pdef
            else:
                print('Using a version before 7.0.0')
                from .old_proposal import make_propagator, make_pdef
                self._make_prop= make_propagator
                self._make_pdef = make_pdef
        else:
            raise ValueError(
                'Unknown lepton propagator! Check the config file'
            )

    def __getitem__(self, key):
        if key not in self._pdef_dict.keys():
            self._pdef_dict[key] = self._make_pdef(key)
        if key not in self._prop_dict.keys():
            self._prop_dict[key] = self._make_prop(key, **self._kwargs)
        return self._pdef_dict[key], self._prop_dict[key]

    def _new_proposal_setup(self):
        """Set up a proposal propagator for version > 7"""
        print('----------------------------------------------------')
        print('Setting up proposal')
        self.args = {}
        print('Using proposal')
        from .new_proposal import _make_medium, make_pdef
        self.__pp_version = pp.__version__
        print('The proposal version is ' + str(self.__pp_version))
        # TODO Make so that lepton is tied to the LI situation / set by it
        self.args['p_def'] = make_pdef(config['lepton propagator']['lepton'])
        self.args['target'] = _make_medium(config['lepton propagator']['medium'])
        self.args['ecut'] = config['lepton propagator']['ecut'][1]
        self.args['vcut'] = config['lepton propagator']['vcut'][1]
        self.args['soft_losses'] = (
            config["lepton propagator"]['soft_losses']
        )
        args = {
            "particle_def": self.args['p_def'],
            "target": self.args['target'],
            "interpolate": config['lepton propagator']['interpolation'],
            "cuts": pp.EnergyCutSettings(
                self.args['ecut'], self.args['vcut'],
                self.args['soft_losses']
            ),
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

    def energy_losses(self, particle, detector_center):
        pdef, prop = self[str(particle)]
        losses = energy_losses(
            prop,
            pdef,
            particle,
            self._kwargs["propagation padding"],
            # TODO get this out of here !!!!!!!!
            self._kwargs["r_detector"]+1000,
            detector_center
        )
        return losses
