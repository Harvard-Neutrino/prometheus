# hebe_ui.py
# David Kim

from config import config
import detector_dictionaries as dd
import geo_utils as gu
import warnings

cpath = config['lepton injector']['simulation']
ylist = ['yes','y']; nlist = ['no','n']


def injector_q():
    use_li = input("\nUse existing injection? yes/no: ")

    if use_li.lower() in ylist:
        injFile = input('File: ')
        config['lepton injector']['use existing injection'] = True
    elif use_li.lower() not in nlist:
        print('invalid input')
        injector_q()

def ranged_q():
    inj_type = input('\nUse ranged injection (else volume injection)? yes/no: ')

    if inj_type in ylist:
        cpath['is ranged'] = True
    elif inj_type not in nlist:
        print('invalid input')
        ranged_q()

def detector_q():
    """
    Sets detector file and calculates selection volume
    """
    dname = input(
    '''
Which detector do you want to use?
- IceCube
- ORCA
- User supplied geo file (UF)\nChoose icecube/orca/uf: '''
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
    else:
        print('invalid input')
        detector_q()

def dfile_q():

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

    # except:
    #     print('File not found')
    #     dfile_q()

def useRec_q():
    use_rec= input('Use reccomended selection volume? yes/no: ')
    
    if use_rec in ylist:
        cpath['injection radius'] = dd.inj_radius
        cpath['endcap length'] = dd.endcap_len
        cpath['cylinder radius'] = dd.cylRadius
        cpath['cylinder height'] = dd.cylHeight
    elif use_rec in nlist:
        warnings.warn("Changing selection volume may affect efficiency")
        cpath['injection radius'] = input('Injection radius [m]: ')
        cpath['endcap length'] = input('Endcap length [m]: ')
        cpath['cylinder radius'] = input('Cylinder radius [m]: ')
        cpath['cylinder height'] = input('Cylinder height [m]: ')
    else:
        print('invalid input')
        useRec_q()
    
    print('detector loaded')

def medium_q():
    medium = input('Medium? ice/water: ')
    if medium.lower() in 'ice water':
        cpath['medium'] = medium
    else:
        print('invalid input')
        medium_q()


def event_q():
    global state_key
    e_type = input('\nWhich event type? NuE/NuMu/NuTau/NuEBar/NuMuBar/NuTauBar: ')
    if e_type.lower() in ['nue','numu','nutau','nuebar','numubar','nutaubar']:
        state_key = e_type.lower()+'/'
    else:
        print('invalid input')
        event_q()

def interaction_q():
    global state_key
    i_type = input('Which interaction type CC/NC/GR: ')
    if i_type.lower() in ['cc','nc']:
        state_key += i_type.lower()
        cpath['final state 1'] = dd.final_state[state_key][0]
        cpath['final state 2'] = dd.final_state[state_key][1]
    elif i_type.lower() == 'gr' and state_key == 'nuebar/':
        gr_q()
    else:
        print('invalid input')
        interaction_q()

def gr_q():
    global state_key
    gr_final = input('Final type? hadron/e/mu/tau: ')
    if gr_final.lower() in ['hardon','e','mu','tau']:
        state_key += gr_final.lower()
        cpath['final state 1'] = dd.final_state[state_key][0]
        cpath['final state 2'] = dd.final_state[state_key][1]
    else:
        print('invalid input')
        gr_q()

def misc_q():
    n_events = input('\nNumber of events: ')
    min_e = input('Minimal energy [GeV]: ')
    max_e = input('Maximal energy [GeV]: ')
    min_zen = input('Min zenith [deg]: ')
    max_zen = input('Miax zenith [deg]: ')
    power_law = input('Set power law: ')

    cpath['nevents'] = float(n_events)
    cpath['minimal energy'] = float(min_e)
    cpath['maximal energy'] = float(max_e)
    cpath['minZenith'] = float(min_zen)
    cpath['maxZenith'] = float(max_zen)
    cpath['power law'] = float(power_law)

def run():
    print('\n=== Prometheus Config ===')
    injector_q()
    ranged_q()
    detector_q()
    event_q()
    interaction_q()
    misc_q()
    print('\ngenerating config file')
    dd.out_doc(config)
    print('\nconfig set!\n')

run()
