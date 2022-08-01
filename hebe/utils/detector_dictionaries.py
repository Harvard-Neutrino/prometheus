# detector_dictionaries.py
# Authors: David Kim
# Values and dicts for hebe_ui

# Vaules for default detectors
detectors = {
    'icecube': {
        'file name': './data/icecube-f2k',
        'injection radius': 900,
        'endcap length': 900,
        'cylinder radius': 700,
        'cylinder height': 1000,
        'medium': 'ice'
    },

    'pone': {
        'file name': './data/pone_triangle-f2k',
        'injection radius': 651,
        'endcap length': 264,
        'cylinder radius': 208,
        'cylinder height': 1300,       
        'medium': 'water'
    }
}

# Table of event & interaction types to final states
final_state = {
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
    'nuebar/mu':['MuMinus','NuMubar'],
    'nuebar/tau':['TauMinus','NuTaubar']
}

    

