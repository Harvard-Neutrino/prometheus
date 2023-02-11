# -*- coding: utf-8 -*-
# hebe_ui.py
# Authors: David Kim
# User interface for Prometheus

from .. import config
from .detector_dictionaries import detectors, final_states
from .geo_utils import (
    from_geo,get_volume,
    get_endcap,
    get_injection_radius
)
from os.path import exists
from colorama import Fore, Style

ylist = ['yes','y']; nlist = ['no','n']

def injector_q():
    use_li = input("Use LeptonInjector? (yes/no): ")

    if use_li.lower() in ylist:
        config['lepton injector']['inject'] = True
    elif use_li.lower() in nlist:
        injfile_q()
        config['lepton injector']['inject'] = False
    elif use_li == '':
        print(f'Loaded defualt value \'{config["lepton injector"]["inject"]}\'')
        if not config["lepton injector"]["inject"]:
            injfile_q()
    else:
        print('invalid input')
        injector_q()

def injfile_q():
    injFile = input('File: ')
    if exists(injFile):
        config['lepton injector']['simulation']['output name'] = injFile
        config['lepton injector']['inject'] = False
    else:
        print(f'File not found: No such fle or directory: \'{injFile}\'')
        injfile_q()

def detector_q():
    """ Sets detector file, medium, and selection volume """
    dname = input(
    '''
(0) User supplied geo file  (3) P-ONE
(1) IceCube                 (4) GVD
(2) IceCube-Gen2
Which detector do you want to use? (0/1/2/3/4): '''
    )

    if str(dname) in detectors:
        dpath = detectors[str(dname)]
        config['lepton injector']['simulation']['injection radius'] = dpath['injection radius']
        config['lepton injector']['simulation']['endcap length'] = dpath['endcap length']
        config['lepton injector']['simulation']['cylinder radius'] = dpath['cylinder radius']
        config['lepton injector']['simulation']['cylinder height'] = dpath['cylinder height']
        config['detector']['medium'] = dpath['medium']
        config['lepton propagator']['medium'] = dpath['medium']
        config['detector']['detector specs file'] = dpath['file path']
        config['lepton injector']['force injection params'] = True
        config['lepton propagator']['force propagation params'] = True
        print(dpath['detector name']+' loaded')
    elif str(dname) == '0':
        dfile_q()
        config['lepton injector']['force injection params'] = True
        config['lepton propagator']['force propagation params'] = True
    elif dname == '':
        print((f'Loaded default value \'{config["detector"]["detector specs file"]}\''))
    elif dname == '..':
        print('')
        injector_q()
        detector_q()
    else:
        print('invalid input')
        detector_q()

def dfile_q():
    try:
        dfile = input('File: ')
        if dfile == '..':
            detector_q()
        else: 
            print('Reading...')
            d_coords,keys,medium = from_geo(dfile)
            d_cyl = get_volume(d_coords)
            config['detector']['medium'] = medium
            config['lepton propagator']['medium'] = medium
            config['detector']['detector specs file'] = dfile
            is_ice = medium.lower()=='ice'
            config['lepton injector']['simulation']['injection radius'] = round(get_injRadius(d_coords, is_ice))
            config['lepton injector']['simulation']['endcap length'] = round(get_endcap(d_coords, is_ice))
            config['lepton injector']['simulation']['cylinder radius'] = round(d_cyl[0])
            config['lepton injector']['simulation']['cylinder height'] = round(d_cyl[1])

            print(f'\nReccomended selection volume:\n  Injection radius: {config["lepton injector"]["simulation"]["injection radius"]} m')
            print(f'  Endcap length: {config["lepton injector"]["simulation"]["endcap length"]} m')
            print(f'  Cylinder radius: {config["lepton injector"]["simulation"]["cylinder radius"]} m')
            print(f'  Cylinder height: {config["lepton injector"]["simulation"]["cylinder height"]} m')
            useRec_q()
    except FileNotFoundError:
        print(f'File not found: No such fle or directory: \'{dfile}\'')
        dfile_q()

def useRec_q():
    use_rec= input('Use reccomended selection volume? (yes/no): ')
    
    if use_rec in nlist:
        print("WARNING: Changing selection volume may affect simulation efficiency")
        set_misc('Injection radius [m]: ', config['lepton injector']['simulation']['injection radius'])
        set_misc('Endcap length [m]: ', config['lepton injector']['simulation']['endcap length'])
        set_misc('Cylinder radius [m]: ', config['lepton injector']['simulation']['cylinder radius'])
        set_misc('Cylinder height [m]: ', config['lepton injector']['simulation']['cylinder height'])
    elif use_rec in ['']+ylist:
        print('Loaded default values')
    elif use_rec == '..':
        dfile_q()
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
        type_index = val_list.index(
            [config['lepton injector']['simulation']['final state 1'], config['lepton injector']['simulation']['final state 2']]
        )
        state_key = key_list[type_index]
        print(f'Loaded default value \'{key_list[type_index]}\'')
        ranged_q()
    elif e_type == '..':
        detector_q()
        event_q()
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
        config['lepton injector']['simulation']['final state 1'] = final_states[state_key][0]
        config['lepton injector']['simulation']['final state 2'] = final_states[state_key][1]
        ranged_q()
    elif i_type.lower() == 'gr' and state_key == 'nuebar/':
        gr_q()
    elif i_type == '':
        key_list = list(final_states.keys())
        val_list = list(final_states.values())
        type_index = val_list.index(
            [config['lepton injector']['simulation']['final state 1'], config['lepton injector']['simulation']['final state 2']]
        )
        print(f'Loaded default value \'{key_list[type_index]}\'')
        ranged_q()
    elif i_type == '..':
        event_q()
    else:
        print('invalid input')
        interaction_q()

def gr_q():
    """ Additional question for glashow resonance """
    global state_key
    gr_final = input('Final type? (Hadron/E/Mu/Tau): ')
    if gr_final.lower() in ['hadron','e','mu','tau']:
        state_key += gr_final.lower()
        config['lepton injector']['simulation']['final state 1'] = final_states[state_key][0]
        config['lepton injector']['simulation']['final state 2'] = final_states[state_key][1]
        ranged_q()
    elif gr_final == '':
        state_key += 'hadron'
        config['lepton injector']['simulation']['final state 1'] = final_states[state_key][0]
        config['lepton injector']['simulation']['final state 2'] = final_states[state_key][1]
        print('Loaded default value \'Hadron\'')
        ranged_q()
    elif gr_final == '..':
        interaction_q()
    else:
        print('invalid input')
        gr_q()

def ranged_q():
    global state_key
    inj_type = input('\nUse ranged injection? (yes/no): ')
    if inj_type in ylist:
        config['lepton injector']['simulation']['is ranged'] = True
        config['run']['group name'] = 'RangedInjector0'
    elif inj_type in nlist:
        config['lepton injector']['simulation']['is ranged'] = False
        config['run']['group name'] = 'VolumeInjector0'
    elif inj_type == '':
        print(f'Loaded defualt value \'{config["lepton injector"]["simulation"]["is ranged"]}\'')
    elif inj_type == '..':
        event_q()
    else:
        print('invalid input')
        ranged_q()
    
    if not config['lepton injector']['simulation']['is ranged'] and state_key == 'numu/cc':
        print('WARNING: Ranged injection is reccomended for NuMu/CC')

class MiscQ():
    def __init__(self, input_str, path, prev_q, next_q = None, is_float = True):
        self.input_str = input_str
        self.path = path
        self.prev_q = prev_q
        self.next_q = next_q
        self.is_float = is_float
   
    def set_misc(self):
        val = input(self.input_str)
        if val == '':
            print(f'Loaded defualt value \'{config["lepton injector"]["simulation"][self.path]}\'')
            if self.next_q != None:
                self.next_q.set_misc()
        elif val == '..':
            if self.prev_q == None:
                ranged_q()
                misc_q()
            else:
                self.prev_q.set_misc()
        else:
            try:
                if self.is_float:
                    config['lepton injector']['simulation'][self.path] = float(val)
                else:
                    config['lepton injector']['simulation'][self.path] = int(val)
                if self.next_q != None:
                    self.next_q.set_misc()
            except ValueError:
                if self.is_float:
                    print(f'ValueError: could not convert string to float: \'{val}\'')
                else:
                    print(f'ValueError: could not convert string to int: \'{val}\'')
                self.set_misc()

def misc_q():
    print('')
    n_events = MiscQ('Number of events: ', 'nevents', None, is_float = False)
    min_e = MiscQ('Min energy [GeV]: ', 'minimal energy', n_events)
    max_e = MiscQ('Max energy [GeV]: ', 'maximal energy', min_e)
    min_z = MiscQ('Min zenith [deg]: ', 'minZenith', max_e)
    max_z = MiscQ('Max zenith [deg]: ', 'maxZenith', min_z)
    p_law = MiscQ('Set power law: ', 'power law', max_z)
    n_events.next_q = min_e
    min_e.next_q = max_e
    max_e.next_q = min_z
    min_z.next_q = max_z
    max_z.next_q = p_law
    n_events.set_misc()

def run_ui():
    """ Runs user interface """
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
    misc_q()
    print('\nconfig set!')
    print('-------------------------------------------')
    return config
