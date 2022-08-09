# hebe_ui.py
# Authors: David Kim
# User interface for Prometheus

from ..config import config
from .detector_dictionaries import detectors, final_states
from .geo_utils import from_geo,get_volume,get_endcap,get_injRadius
from warnings import warn
from colorama import Fore, Style

cpath = config['lepton injector']['simulation']
ylist = ['yes','y']; nlist = ['no','n']

def injector_q():
    use_li = input("Use LeptonInjector? (yes/no): ")

    if use_li.lower() in ylist:
        config['lepton injector']['inject'] = True
    elif use_li.lower() in nlist:
        injFile = input('File: ')
        config['lepton injector']['inject'] = False
    elif use_li == '':
        print(f'Loaded defualt value \'{config["lepton injector"]["inject"]}\'')
    elif use_li.lower() not in nlist:
        print('invalid input')
        injector_q()

def detector_q():
    """Sets detector file, medium, and selection volume
    """
    dname = input(
    '''
Which detector do you want to use?
- IceCube 
- P-ONE
- User supplied geo file (UF)\n(icecube/pone/uf): '''
    )

    if dname.lower() in detectors:
        dpath = detectors[dname.lower()]
        cpath['injection radius'] = dpath['injection radius']
        cpath['endcap length'] = dpath['endcap length']
        cpath['cylinder radius'] = dpath['cylinder radius']
        cpath['cylinder height'] = dpath['cylinder height']
        config['detector']['medium'] = dpath['medium']
        config['lepton propagator']['medium'] = dpath['medium']
        config['detector']['detector specs file'] = dpath['detector specs file']
        print(dname.lower()+' loaded')
    elif dname.lower() == 'uf':
        dfile_q()
        print(f'\nReccomended selection volume:\n  Injection radius: {cpath["injection radius"]} m')
        print(f'  Endcap length: {cpath["endcap length"]} m\n  Cylinder radius: {cpath["cylinder radius"]} m')
        print(f'  Cylinder height: {cpath["cylinder height"]} m')
        useRec_q()
    elif dname == '':
        print((f'Loaded default value \'{config["detector"]["detector specs file"]}\''))
    else:
        print('invalid input')
        detector_q()

def dfile_q():
    try:
        dfile = input('File: ')
        print('Reading...')
        d_coords,keys,medium = from_geo(dfile)
        d_cyl = get_volume(d_coords)
        config['detector']['medium'] = medium
        config['lepton propagator']['medium'] = medium
        config['detector']['detector specs file'] = dfile
        is_ice = medium.lower()=='ice'
        cpath['injection radius'] = round(get_injRadius(d_coords, is_ice))
        cpath['endcap length'] = round(get_endcap(d_coords, is_ice))
        cpath['cylinder radius'] = round(d_cyl[0])
        cpath['cylinder height'] = round(d_cyl[1])

    except FileNotFoundError:
        print(f'File not found: No such fle or directory: \'{dfile}\'')
        dfile_q()

def useRec_q():
    use_rec= input('Use reccomended selection volume? (yes/no): ')
    
    if use_rec in nlist:
        warn("Changing selection volume may affect efficiency")
        set_misc('Injection radius [m]: ', cpath['injection radius'])
        set_misc('Endcap length [m]: ', cpath['endcap length'])
        set_misc('Cylinder radius [m]: ', cpath['cylinder radius'])
        set_misc('Cylinder height [m]: ', cpath['cylinder height'])
    elif use_rec in ['']+ylist:
        print('Loaded default values')
    else:
        print('invalid input')
        useRec_q()

def event_q():
    global state_key
    e_type = input('\nWhich event type? (NuE/NuMu/NuTau/NuEBar/NuMuBar/NuTauBar): ')
    if e_type.lower() in ['nue','numu','nutau','nuebar','numubar','nutaubar']:
        state_key = e_type.lower()+'/'
        interaction_q()
    elif e_type == '':
        key_list = list(final_states.keys())
        val_list = list(final_states.values())
        type_index = val_list.index([cpath['final state 1'], cpath['final state 2']])
        state_key = key_list[type_index]
        print(f'Loaded default value \'{key_list[type_index]}\'')
    else:
        print('invalid input')
        event_q()

def interaction_q():
    global state_key
    if state_key == 'nuebar/':
        i_type = input('Which interaction type (CC/NC/GR): ')
    else:
        i_type = input('Which interaction type (CC/NC): ')

    if i_type.lower() in ['cc','nc']:
        state_key += i_type.lower()
        cpath['final state 1'] = final_states[state_key][0]
        cpath['final state 2'] = final_states[state_key][1]
    elif i_type.lower() == 'gr' and state_key == 'nuebar/':
        gr_q()
    elif i_type == '':
        key_list = list(final_states.keys())
        val_list = list(final_states.values())
        type_index = val_list.index([cpath['final state 1'], cpath['final state 2']])
        print(f'Loaded default value \'{key_list[type_index]}\'')
    else:
        print('invalid input')
        gr_q()

def gr_q():
    """Additional question for glashow resonance"""
    global state_key
    gr_final = input('Final type? (Hadron/E/Mu/Tau): ')
    if gr_final.lower() in ['hadron','e','mu','tau']:
        state_key += gr_final.lower()
        cpath['final state 1'] = final_states[state_key][0]
        cpath['final state 2'] = final_states[state_key][1]
    elif gr_final == '':
        state_key += 'hadron'
        cpath['final state 1'] = final_states[state_key][0]
        cpath['final state 2'] = final_states[state_key][1]
        print('Loaded default value \'Hadron\'')
    else:
        print('invalid input')
        gr_q()

def ranged_q():
    global state_key
    inj_type = input('\nUse ranged injection? (yes/no): ')
    if inj_type in ylist:
        cpath['is ranged'] = True
        config['run']['group name'] = 'RangedInjector0'
    elif inj_type in nlist:
        cpath['is ranged'] = False
        config['run']['group name'] = 'VolumeInjector0'
    elif inj_type == '':
        print(f'Loaded defualt value \'{cpath["is ranged"]}\'')
    else:
        print('invalid input')
        ranged_q()
    
    if not cpath['is ranged'] and state_key == 'numu/cc':
        warn('Ranged injection is reccomended for NuMu/CC')

def misc_q():
    set_misc('\nNumber of events: ', cpath['nevents'])
    set_misc('Min energy [GeV]: ', cpath['minimal energy'])
    set_misc('Max energy [GeV]: ', cpath['maximal energy'])
    set_misc('Min zenith [deg]: ', cpath['minZenith'])
    set_misc('Max zenith [deg]: ', cpath['maxZenith'])
    set_misc('Set power law: ', cpath['power law'])

def set_misc(input_str, path):
    val = input(input_str)
    if val == '':
        print(f'Loaded defualt value \'{path}\'')
    else:
        path = float(val)

def run_ui():
    """Runs user interface"""
    print(f"""
——————————————————————————————————————————————————————————
Welcome to
{Fore.RED} ____                           _   _                    
{Fore.YELLOW}|  _ \ _ __ ___  _ __ ___   ___| |_| |__   ___ _   _ ___ 
{Fore.GREEN}| |_) | '__/ _ \| '_ ` _ \ / _ \ __| '_ \ / _ \ | | / __|
{Fore.BLUE}|  __/| | | (_) | | | | | |  __/ |_| | | |  __/ |_| \__ \\
{Fore.MAGENTA}|_|   |_|  \___/|_| |_| |_|\___|\__|_| |_|\___|\__,_|___/{Style.RESET_ALL}
——————————————————————————————————————————————————————————
""")
    injector_q()
    detector_q()
    event_q()
    ranged_q()
    misc_q()
    print('\nconfig set!')
    print('-------------------------------------------')
    return config