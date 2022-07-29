# hebe_ui.py
# Authors: David Kim
# User interface for Prometheus

from config import config
import detector_dictionaries as dd
import geo_utils as gu
from warnings import warn

cpath = config['lepton injector']['simulation']
ylist = ['yes','y']; nlist = ['no','n']
state_key =''

def injector_q():
    use_li = input("\nUse existing injection? yes/no: ")

    if use_li.lower() in ylist:
        injFile = input('File: ')
        config['lepton injector']['use existing injection'] = True
    elif use_li == '':
        load_default(config['lepton injector']['use existing injection'])
    elif use_li.lower() not in nlist:
        print('invalid input')
        injector_q()

def detector_q():
    """Sets detector file and calculates selection volume
    """
    dname = input(
    '''
Which detector do you want to use?
- IceCube
- P-ONE
- User supplied geo file (UF)\nChoose icecube/pone/uf: '''
    )

    if dname.lower() in dd.detectors:
        dpath = dd.detectors[dname.lower()]
        cpath['injection radius'] = dpath['injection radius']
        cpath['endcap length'] = dpath['endcap length']
        cpath['cylinder radius'] = dpath['cylinder radius']
        cpath['cylinder height'] = dpath['cylinder height']
        cpath['medium'] = dpath['medium']
        config['detector']['file name'] = dpath['file name']
        print(dname.lower()+' loaded')
    elif dname.lower() == 'uf':
        dfile_q()
        print(f'\nReccomended selection volume:\n\tInjection radius: {dd.inj_radius} m')
        print(f'\tEndcap length: {dd.endcap_len} m\n\tCylinder radius: {dd.cylRadius} m')
        print(f'\tCylinder height: {dd.cylHeight} m')
        useRec_q()
    elif dname == '':
        load_default(config['detector']['file name'])
    else:
        print('invalid input')
        detector_q()

def dfile_q():
    try:
        dfile = input('File: ')
        is_ice = True
        d_coords,keys,medium = gu.from_geo(dfile)
        d_cyl = gu.get_cylinder(d_coords)
        config['lepton propagator']['medium'] = medium
        config['detector']['file name'] = dfile

        if medium.lower() == "ice":
            padding = gu.ice_padding
        else:
            padding = gu.water_padding
            is_ice = False
        dd.cylRadius = round(d_cyl[0]+padding)
        dd.cylHeight = round(d_cyl[1]+2*padding)
        dd.endcap_len = round(gu.get_endcap(d_coords, is_ice=is_ice))
        dd.inj_radius = round(gu.get_injRadius(d_coords, is_ice=is_ice))

    except FileNotFoundError:
        print('File not found')
        dfile_q()

def useRec_q():
    use_rec= input('Use reccomended selection volume? yes/no: ')
    
    if use_rec in ylist:
        cpath['injection radius'] = dd.inj_radius
        cpath['endcap length'] = dd.endcap_len
        cpath['cylinder radius'] = dd.cylRadius
        cpath['cylinder height'] = dd.cylHeight
    elif use_rec in nlist:
        warn("Changing selection volume may affect efficiency")
        cpath['injection radius'] = input('Injection radius [m]: ')
        cpath['endcap length'] = input('Endcap length [m]: ')
        cpath['cylinder radius'] = input('Cylinder radius [m]: ')
        cpath['cylinder height'] = input('Cylinder height [m]: ')
    elif use_rec == '':
        print('Loaded default values')
    else:
        print('invalid input')
        useRec_q()

def medium_q():
    medium = input('Medium? ice/water: ')
    if medium.lower() in 'ice water':
        cpath['medium'] = medium
    else:
        print('invalid input')
        medium_q()


def event_q():
    e_type = input('\nWhich event type? NuE/NuMu/NuTau/NuEBar/NuMuBar/NuTauBar: ')
    if e_type.lower() in ['nue','numu','nutau','nuebar','numubar','nutaubar']:
        state_key = e_type.lower()+'/'
        interaction_q()
    elif e_type == '':
        key_list = list(dd.final_state.keys())
        val_list = list(dd.final_state.values())
        type_index = val_list.index([cpath['final state 1'], cpath['final state 2']])
        print(f'Loaded default value {key_list[type_index]}')
    else:
        print('invalid input')
        event_q()

def interaction_q():
    i_type = input('Which interaction type CC/NC/GR: ')
    if i_type.lower() in ['cc','nc']:
        state_key += i_type.lower()
        cpath['final state 1'] = dd.final_state[state_key][0]
        cpath['final state 2'] = dd.final_state[state_key][1]
    elif i_type.lower() == 'gr' and state_key == 'nuebar/':
        gr_q()
    elif i_type == '':
        key_list = list(dd.final_state.keys())
        val_list = list(dd.final_state.values())
        type_index = val_list.index([cpath['final state 1'], cpath['final state 2']])
        print(f'Loaded default value {key_list[type_index]}')
    else:
        print('invalid input')
        interaction_q()

def ranged_q():
    inj_type = input('\nUse ranged injection (else volume injection)? yes/no: ')
    if inj_type in ylist:
        cpath['is ranged'] = True
    elif inj_type == '':
        load_default(cpath['is ranged'])
    elif inj_type not in nlist:
        print('invalid input')
        ranged_q()
    
    if not cpath['is ranged'] and state_key == 'numu/cc':
        warn('Ranged injection is reccomended for NuMu/CC')

def gr_q():
    gr_final = input('Final type? hadron/e/mu/tau: ')
    if gr_final.lower() in ['hardon','e','mu','tau']:
        state_key += gr_final.lower()
        cpath['final state 1'] = dd.final_state[state_key][0]
        cpath['final state 2'] = dd.final_state[state_key][1]
    else:
        print('invalid input')
        gr_q()

def misc_q():
    set_misc('\nNumber of events: ', cpath['nevents'])
    set_misc('Minimal energy [GeV]: ', cpath['minimal energy'])
    set_misc('Maximal energy [GeV]: ', cpath['maximal energy'])
    set_misc('Min zenith [deg]: ', cpath['minZenith'])
    set_misc('Max zenith [deg]: ', cpath['maxZenith'])
    set_misc('Set power law: ', cpath['power law'])

def load_default(path):
    print(f'Loaded defualt value {path}')

def set_misc(input_str, path):
    val = input(input_str)
    if val == '':
        load_default(path)
    else:
        path = float(val)

def run():
    """Runs user interface"""
    print('\n=== Prometheus Config ===')
    injector_q()
    detector_q()
    event_q()
    ranged_q()
    misc_q()
    # print('\ngenerating config file')
    # dd.out_doc(config)
    print('\nconfig set!\n')

run()
