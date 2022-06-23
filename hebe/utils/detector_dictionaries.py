# detector_dictionaries.py
# David Kim

from .config import config

# default vaules for detectors
# actually calculate for orca
detectors = {
    'icecube': {
        'file name': '../hebe/data/icecube-f2k',
        'injection radius': 900,
        'endcap length': 900,
        'cylinder radius': 700,
        'cylinder height': 1000,
        'medium': 'ice'
    },

    'orca': {
        'file name': '../hebe/data/pone_triangle-f2k',
        'injection radius': 900,
        'endcap length': 900,
        'cylinder radius': 700,
        'cylinder height': 1000,       
        'medium': 'water'
    }
}

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
    'nutaubar/nc':['NuTauBar','Hadrons']
}

# placeholders
injRadius = 900
endLength = 900
cylRadius = 700
cylHeight = 1000

# move this somewhere else
def out_doc(cdict):
    out = open('config_settings.txt','w')

    for key in cdict:
        out.write('\n'+key+': \n')
        for param in cdict[key]:
            if isinstance(cdict[key][param], dict):
                out.write('  '+param+': \n')
                for val in cdict[key][param]:
                    out.write('    '+val+': '+str(cdict[key][param][val])+'\n')
            else:
                out.write('  '+param+': '+str(cdict[key][param])+'\n')
    out.close()

