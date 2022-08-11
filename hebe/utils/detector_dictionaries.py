# -*- coding: utf-8 -*-
# detector_dictionaries.py
# Authors: David Kim
# Dicts for hebe_ui

# Vaules for default detectors
detectors = {
    '1': {
        'detector name': 'IceCube',
        'file path': '../hebe/data/icecube-geo',
        'injection radius': 900,
        'endcap length': 900,
        'cylinder radius': 700,
        'cylinder height': 1000,
        'medium': 'ice'
    },

    '2': {
        'detector name': 'IceCube-Gen2',
        'file path': '../hebe/data/icecube_gen2-geo',
        'injection radius': 2100,
        'endcap length': 1400,
        'cylinder radius': 2000,
        'cylinder height': 1700,
        'medium': 'ice'
    },

    '3': {
        'detector name': 'P-ONE',
        'file path': '../hebe/data/pone_triangle-geo',
        'injection radius': 650,
        'endcap length': 300,
        'cylinder radius': 200,
        'cylinder height': 1300,       
        'medium': 'water'
    },

    '4': {
        'detector name': 'GVD',
        'file path': '../hebe/data/gvd-geo',
        'injection radius': 500,
        'endcap length': 300,
        'cylinder radius': 300,
        'cylinder height': 900,       
        'medium': 'water'
    }
}

# Table of event & interaction types to final states
final_states = {
    'nue/cc':['EMinus','Hadrons'],
    'numu/cc':['MuMinus','Hadrons'],
    'nutau/cc':['TauMinus','Hadrons'],
    'nue/nc':['NuE','Hadrons'],
    'numu/nc':['NuMu','Hadrons'],
    'nutau/nc':['NuTau','Hadrons'],
    
    'nuebar/cc':['EPlus','Hadrons'],
    'numubar/cc':['MuPlus','Hadrons'],
    'nutaubar/cc':['TauPlus','Hadrons'],
    'nuebar/nc':['NuEBar','Hadrons'],
    'numubar/nc':['NuMuBar','Hadrons'],
    'nutaubar/nc':['NuTauBar','Hadrons'],
    
    'nuebar/hadron':['Hadrons','Hadrons'],
    'nuebar/e':['EMinus','NuEBar'],
    'nuebar/mu':['MuMinus','NuMuBar'],
    'nuebar/tau':['TauMinus','NuTauBar']
}

    

