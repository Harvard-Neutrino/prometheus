# -*- coding: utf-8 -*-
# Name: config.py
# Copyright (C) 2022 Stephan Meighen-Berger
# Config file for the pr_dformat package.

from typing import Dict, Any
import yaml
import os

RESOURCES_DIR = os.path.abspath(f"{os.path.dirname(__file__)}/../resources/")

_baseconfig: Dict[str, Any]

_baseconfig = {
    ###########################################################################
    # General inputs
    ###########################################################################
    "general": {
        # Random state seed
        "version": "github"
    },
    ###########################################################################
    # Scenario input
    ###########################################################################
    "run": {
        "run number": 1337,
        'nevents': 10,
        'storage prefix': './output/',
        'config name': 'config',
        'full output' : False,
        # Random seed will follow run number if None
        "random state seed": None,
        'subset': None,
    },
    ###########################################################################
    # Detector
    ###########################################################################
    "detector": {
        'geo file': None,  # Name of the file to use for build
    },
    ###########################################################################
    # Injection
    ###########################################################################
    'injection': {
        'name': 'LeptonInjector',
        'LeptonInjector': {
            'inject': True,
            'paths':{
                'install location': '/opt/LI/install/lib/python3.9/site-packages',
                'xsec dir': f'{RESOURCES_DIR}/cross_section_splines/',
                # These fields will be set with output prefix and run number
                "earth model location": None,
                'injection file': None,
                "lic file": None,
                'diff xsec': None,
                'total xsec': None,
            },
            'simulation': {
                'is ranged': False,
                'final state 1': 'MuMinus',
                'final state 2': 'Hadrons',
                'minimal energy': 1e3, # GeV
                'maximal energy': 1e6, # GeV
                'power law': 1.0,
                'min zenith': 0.0, # degree
                'max zenith': 100.0, # degree
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
                'injection file': None,
            },
            'simulation': {}
        },
        'GENIE':{
            'inject': False,
            'paths': {
                "injection file": None,
            },
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
                "tables path": "~/.local/share/PROPOSAL/tables",
                "earth model location": None,
            },
            "simulation":{
                'track length': 5000,
                # TODO figure out why this breaks for 1e-2
                'vcut': 1,
                #'vcut': [1e-2, 1e-2],
                'ecut': 0.5, # GeV
                'soft losses': False,
                'propagation padding': 900,
                'interpolation': True,
                'lpm effect': True,
                'continuous randomization': True,
                'soft losses': True,
                "interpolate": True,
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
                "tables path": "~/.local/share/PROPOSAL/tables",
                "earth model location": None,
            },
            "simulation":{
                'track length': 5000,
                'vcut': 1,
                'ecut': 0.1, # GeV
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
                "photon field name": "photons",
                "outfile": None
            },
            "simulation": {
                'files': True,
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
                'force': False,
                "ppc_tmpdir:": "./.ppc_tmp",
                'ppc_tmpfile':'.event_hits.ppc.tmp',
                'f2k_tmpfile':'.event_losses.f2k.tmp',
                'ppc_prefix':'',
                'f2k_prefix':'',
                'ppctables':'../resources/PPC_tables/ic_accept_all/',
                'ppc_exe':'../resources/PPC_executables/PPC_CUDA/ppc', # binary executable
                "photon field name": "photons",
                "outfile": None
            },
            "simulation": {
                'device':0, # GPU,
                'supress_output': True,
            }
        },
        'PPC': {
            "paths": {
                'location': '../PPC/',
                'force': False,
                "ppc_tmpdir:": "./.ppc_tmp",
                'ppc_tmpfile': '.event_hits.ppc.tmp',
                'f2k_tmpfile': '.event_losses.f2k.tmp',
                'ppc_prefix':'',
                'f2k_prefix':'',
                'ppctables':'../resources/PPC_tables/ic_accept_all/',
                'ppc_exe': '../resources/PPC_executables/PPC/ppc',  # binary executable
                "photon field name": "photons",
                "outfile": None,
            },
            "simulation": {
                'device': 0,  # CPU
                'supress_output': True
            }
        },
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
