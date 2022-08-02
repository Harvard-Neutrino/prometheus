# -*- coding: utf-8 -*-
# Name: config.py
# Authors: Stephan Meighen-Berger
# Config file for the pr_dformat package.

from typing import Dict, Any
import yaml

_baseconfig: Dict[str, Any]

_baseconfig = {
    ###########################################################################
    # General inputs
    ###########################################################################
    "general": {
        # Random state seed
        "version": "GitHub",
        "random state seed": 1337,
        'config location': '../run/config.json',
        'meta data file': True,  # Switch to store meta data file
        'meta name': 'meta_data',
        'clean up': False,  # Delete all intermediate and temporary files
        'full output' : False
    },
    ###########################################################################
    # Scenario input
    ###########################################################################
    "run": {
        # Defines some run parameters
        'group name': 'VolumeInjector0',
        # The different data sets. The boolean denotes the final state sets
        'data sets': [
            (False, 'initial'),
            (False, 'properties'),
            (True, 'final_1'),
            (True, 'final_2')
        ],
        # If a subset should be used:
        'subset': {
            'switch': False,
            'counts': 10,
        },
        'noise': False,
    },
    ###########################################################################
    # Detector
    ###########################################################################
    "detector": {
        'new detector': False,  # Flag to generate a new detector file
        'detector specs file': 'unnamed',  # Name of the file to use for build
        # Padding for sphere where we do physics good
        'padding' : 200, # m
        'radius' : 900, # m
        'r_max' : 1e18, # m
        "medium" : "ice"
    },
    ###########################################################################
    # Paricles
    ###########################################################################
    'particles':
    {
        # Track particles
        'track particles': [13, -13],
        # Everything here will be treated explicitly
        'explicit': [11, -11, 111, 211, 13, -13, 15, -15],
        # Everything else is replaced by
        'replacement': 2212,
    },
    ###########################################################################
    # Lepton injector
    ###########################################################################
    'lepton injector': {
        'inject': True,
        'location': '/opt/LI/install/lib/python3.9/site-packages',
        'xsec location': '/opt/LI/source/resources/',
        'simulation': {
            'nevents': 10,
            'diff xsec': "/test_xs.fits",
            'total xsec': "/test_xs_total.fits",
            'is ranged': False,
            'final state 1': 'MuMinus',
            'final state 2': 'Hadrons',
            'minimal energy': 1e3,  # GeV
            'maximal energy': 1e6,
            'power law': 2.,
            'minZenith': 80.,  # deg
            'maxZenith': 180.,
            'minAzimuth': 0.,
            'maxAzimuth': 360.,
            'earth model location': "earthparams/",
            'earth model': "Planet",
            'output name': "./data_output.h5",
            "lic name": "./config.lic",
            "injection radius":900, # m
            "endcap length":900, # m
            "cylinder radius":700, # m
            "cylinder height":1000, # m
        },
        'force injection params': False
    },
    ###########################################################################
    # Lepton propagator
    ###########################################################################
    'lepton propagator': {
        'name': 'proposal',
        'track length': 5000,  # maximum track length to model
        'lepton': 'MuMinus',
        'medium': 'Water',
        # TODO I made these numbers up !!!!!!!!!!
        'vcut': [1, 1],
        'ecut': [100.0, 500.0],  # MeV
        'soft_losses': False,
        'propagation padding': 900,
        'interpolation': True,
        'lpm_effect' : True,
        'continuous_randomization' : True,
        'soft_losses' : True,
        'scattering model' : "Moliere",
        'force propagation params':False

    },
    ###########################################################################
    # Photon propagator
    ###########################################################################
    'photon propagator': {
        'name': 'olympus',
        'storage location': './output/',
        'olympus': {
            'location': '../',
            'files': True,
            # The photon propagation model parameters:
            'photon model': 'pone_config.json',
            'flow': "photon_arrival_time_nflow_params.pickle",
            'counts': "photon_arrival_time_counts_params.pickle",
            'wavelength': 700,  # in nm
        },
        'PPC_CUDA':{
            'location':'../PPC_CUDA/',
            'ppc_tmpfile':'.event_hits.ppc.tmp',
            'f2k_tmpfile':'.event_losses.f2k.tmp',
            'ppc_prefix':'',
            'f2k_prefix':'',
            'ppctables':'../PPC_CUDA/',
            'ppc_exe':'../PPC_CUDA/ppc', # binary executable
            'device':0, # GPU,
            'supress_output': True,
        },
        'PPC': {
            'location': '../PPC/',
            'ppc_tmpfile': '.event_hits.ppc.tmp',
            'f2k_tmpfile': '.event_losses.f2k.tmp',
            'ppctables': '../PPC/',
            'ppc_exe': '../PPC/ppc',  # binary executable
            'device': 0,  # CPU
            'supress_output': True,
        },
    },
    ###########################################################################
    # Plot
    ###########################################################################
    'plot': {
        'xrange': [-1500, 1500],
        'yrange': [-1500, 1500],
        'zrange': [-3000, 100]
    }
}


class ConfigClass(dict):
    """ The configuration class. This is used
    by the package for all parameter settings. If something goes wrong
    its usually here.
    Parameters
    ----------
    config : dic
        The config dictionary
    Returns
    -------
    None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: Update this
    def from_yaml(self, yaml_file: str) -> None:
        """ Update config with yaml file
        Parameters
        ----------
        yaml_file : str
            path to yaml file
        Returns
        -------
        None
        """
        yaml_config = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        self.update(yaml_config)

    # TODO: Update this
    def from_dict(self, user_dict: Dict[Any, Any]) -> None:
        """ Creates a config from dictionary
        Parameters
        ----------
        user_dict : dic
            The user dictionary
        Returns
        -------
        None
        """
        self.update(user_dict)


config = ConfigClass(_baseconfig)