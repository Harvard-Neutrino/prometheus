# detector_dictionaries.py
# David Kim

cylRadius = 1200
cylHeight = 1200
endcap_len = 1200
inj_radius = 1200

# default vaules for detectors
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

# move this somewhere else
def out_doc(cdict):
    with open('config_settings.txt','w') as out:
        for key in cdict:
            out.write('\n'+key+': \n')
            for param in cdict[key]:
                if isinstance(cdict[key][param], dict):
                    out.write('  '+param+': \n')
                    for val in cdict[key][param]:
                        out.write('    '+val+': '+str(cdict[key][param][val])+'\n')
                else:
                    out.write('  '+param+': '+str(cdict[key][param])+'\n')
    

