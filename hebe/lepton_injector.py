# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Jeffrey Lazar, Stephan Meighen-Berger
# Interface class to the different lepton injectors

from .config import config
from math import pi
import os
import sys
sys.path.append(config['lepton injector']['location'])
try:
    import LeptonInjector as LI
except ImportError:
    raise ImportError('Leptoninjector not found!')


class LepInj(object):
    """ Interface class for the lepton injector module
    """
    def __init__(self):
        print('Setting up the LI')
        print('Fetching parameters and setting paths')
        self._loc_config = config['lepton injector']
        xs_folder = os.path.join(
            os.path.dirname(__file__),
            self._loc_config['xsec location']
        )
        self._spars = self._loc_config['simulation']
        print('Setting the simulation parameters for the LI')
        n_events = self._spars['nevents']
        diff_xs = xs_folder + self._spars['diff xsec']
        total_xs = xs_folder + self._spars['total xsec']
        is_ranged = self._spars['is ranged']
        particles = []
        for id_name, names in enumerate([
                self._spars['final state 1'], self._spars['final state 2']
                ]):
            particles.append(getattr(LI.Particle.ParticleType, names))
        
        print('Setting up the LI object')
        the_injector = LI.Injector(
            n_events, particles[0], particles[1], diff_xs, total_xs, is_ranged
        )
        print('Setting injection parameters')
        deg = pi / 180.

        # define some defaults
        minE = self._spars['minimal energy']     # [GeV]
        maxE = self._spars['maximal energy']    # [GeV]
        gamma = self._spars['power law']
        minZenith = self._spars['minZenith'] * deg
        maxZenith = self._spars['maxZenith'] * deg
        minAzimuth = self._spars['minAzimuth'] * deg
        maxAzimuth = self._spars['maxAzimuth'] * deg
        injectRad = self._spars["injection radius"]
        endcapLength = self._spars["endcap length"]
        cylRad = self._spars["cylinder radius"]
        cylHeight = self._spars["cylinder height"]
        print('Building the injection handler')
        # construct the controller
        if is_ranged:
            controller = LI.Controller(
                the_injector, minE, maxE, gamma, minAzimuth,
                maxAzimuth, minZenith, maxZenith, 
            )
        else:
            controller = LI.Controller(
                the_injector, minE, maxE, gamma, minAzimuth,
                maxAzimuth, minZenith, maxZenith,
                injectRad, endcapLength, cylRad, cylHeight
            )
        print('Defining the earth model')
        # specify the output, earth model
        path_to = os.path.join(
            os.path.dirname(__file__),
            xs_folder, self._spars['earth model location']
        )
        print(
            'Earth model location to use: ' +
            path_to + ' With the model ' + self._spars['earth model']
        )
        print(self._spars['earth model'], self._spars['earth model location'])
        # controller.SetEarthModel(
        #     self._spars['earth model'],
        #     self._spars['earth model location']
        # )
        controller.SetEarthModel(self._spars['earth model'], path_to)
        print("Setting the seed")
        controller.setSeed(config["general"]["random state seed"])
        print('Defining the output location')
        controller.NameOutfile(self._spars['output name'])
        controller.NameLicFile(self._spars['lic name'])

        # run the simulation
        controller.Execute()
