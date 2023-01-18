# -*- coding: utf-8 -*-
# Name: config.py
# Copyright (C) 2022 Stephan Meighen-Berger
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
        'storage location': './output/',
        'full output' : False
    },
    ###########################################################################
    # Scenario input
    ###########################################################################
    "run": {
        # Defines some run parameters
        'nevents': 10,
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
        'specs file': None,  # Name of the file to use for build
    },
    # Injection
    ###########################################################################
    'injection': {
        'name': 'LeptonInjector',
        'LeptonInjector': {
            'inject': True,
            'paths':{
                'install location': '/opt/LI/install/lib/python3.10/site-packages',
                'xsec location': '/opt/LI/source/resources/',
                'diff xsec': "/test_xs.fits",
                'total xsec': "/test_xs_total.fits",
                'output name': "./data_output.h5",
                "lic name": "./config.lic",
                'earth model location': "earthparams/",
            },
            'simulation': {
                'is ranged': False,
                'final state 1': 'MuMinus',
                'final state 2': 'Hadrons',
                'minimal energy': 1e3, # GeV
                'maximal energy': 1e6, # GeV
                'power law': 1.0,
                'min zenith': 80.0, # degree
                'max zenith': 180.0, # degree
                'min azimuth': 0.0, # degree
                'max azimuth': 360.0, # degree
                'earth model': "Planet",
                # The following None params will be set internally unless specified
                "injection radius": None, # m
                "endcap length": None, # m
                "cylinder radius": None, # m
                "cylinder height": None, # m
            },
        },
        'Prometheus':{
            'inject': False,
            'paths': {
                'output name': "./data_output.parquet"
            },
            'simulation': {}
        },
        'GENIE':{
            'inject': False,
            'paths': {},
            'simulation': {}
        }
    },
    ###########################################################################
    # Lepton propagator
    ###########################################################################
    'lepton propagator': {
        'name': 'new proposal',
        # PROPOSAL with versions >= 7
        "new proposal":{
            "paths":{
                "tables path": "~/.local/share/PROPOSAL/tables"
            },
            "simulation":{
                'track length': 5000,
                # TODO figure out why this breaks for 1e-2
                'vcut': [1, 1],
                #'vcut': [1e-2, 1e-2],
                'ecut': [0.5, 0.5], # GeV
                'soft losses': False,
                'propagation padding': 900,
                'interpolation': True,
                'lpm effect': True,
                'continuous randomization': True,
                'soft losses': True,
                'scattering model': "Moliere",
                'maximum radius': 1e18, # m
                # all none elements will be set off detector config settings
                'inner radius': None,
                'medium': None,
            },
        },
        # PROPOSAL with versions <= 6
        "old proposal":{
            "paths":{
                "tables path": "~/.local/share/PROPOSAL/tables"
            },
            "simulation":{
                'track length': 5000,
                'vcut': [1, 1],
                'ecut': [0.1, 0.5], # GeV
                'soft losses': False,
                'propagation padding': 900,
                'interpolation': True,
                'lpm effect': True,
                'continuous randomization': True,
                'soft losses': True,
                'scattering model': "Moliere",
                'force propagation params': False,
                'maximum radius' : 1e18, # m
                # all none elements will be set off detector config settings
                'inner radius': None,
                'medium': None,
            }
        }
    },
    ###########################################################################
    # Photon propagator
    ###########################################################################
    'photon propagator': {
        'name': 'olympus',
        'olympus': {
            "paths": {
                'location': '../resources/',
                'photon model': 'pone_config.json',
                'flow': "photon_arrival_time_nflow_params.pickle",
                'counts': "photon_arrival_time_counts_params.pickle",
            },
            "simulation": {
                'files': True,
                # The photon propagation model parameters:
                'wavelength': 700,  # in nm
                'splitter': 3000,  # Module chunks to work on at once (higher = faster but more memory)
            },
            'particles':
            {
                'track particles': [13, -13],
                'explicit': [11, -11, 111, 211, 13, -13, 15, -15],
                'replacement': 2212,
            },
        },
        'PPC_CUDA':{
            "paths":{
                'location':'../PPC_CUDA/',
                'ppc_tmpfile':'.event_hits.ppc.tmp',
                'f2k_tmpfile':'.event_losses.f2k.tmp',
                'ppc_prefix':'',
                'f2k_prefix':'',
                'ppctables':'../PPC_tables/ic_accept_all/',
                'ppc_exe':'../PPC_CUDA/ppc', # binary executable
            },
            "simulation": {
                'device':0, # GPU,
                'supress_output': True,
            }
        },
        'PPC': {
            "paths": {
                'location': '../PPC/',
                'ppc_tmpfile': '.event_hits.ppc.tmp',
                'f2k_tmpfile': '.event_losses.f2k.tmp',
                'ppc_prefix':'',
                'f2k_prefix':'',
                'ppctables':'../PPC_tables/ic_accept_all/',
                'ppc_exe': '../PPC/ppc',  # binary executable
            },
            "simulation": {
                'device': 0,  # CPU
                'supress_output': True
            }
        },
    },
    ###########################################################################
    # Plot
    ###########################################################################
    # TODO do we wanna keep plotting as an option internally ? I kind think we should
    # totally factor it out
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
