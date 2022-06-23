# hebe_ui.py
# David Kim

from hebe import config
import detector_dictionaries as dd
import f2k_utils as fu

cpath = config['lepton injector']['simulation']
ylist = ['yes','ye','y']; nlist = ['no','n']

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
    dname = input(
    '''
Which detector do you want to use?
- IceCube
- ORCA
- User supplied f2k file (UF)\nChoose icecube/orca/uf: '''
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
        dfile = input('File: ')
        config['detector']['file name'] = dfile

        # compute recc. selection vol
        d_coords = fu.get_xyz(dfile)
        d_cyl = fu.get_cylinder(d_coords)
        dd.cylRadius = round(d_cyl[0]+fu.padding)
        dd.cylHeight = round(d_cyl[1]+2*fu.padding)
        dd.endcap_len = round(fu.get_endcap(d_coords))
        dd.inj_radius = round(fu.get_injRadius(d_coords))

        print('\nReccomended selection volume:\n  Injection radius: '+str(dd.inj_radius)+' m')
        print('  Endcap length: '+str(dd.endcap_len)+' m'+'\n  Cylinder radius: '+str(dd.cylRadius)+' m')
        print('  Cylinder height: '+str(dd.cylHeight)+' m')
        
        useRec_q()
    else:
        print('invalid input')
        detector_q()

def useRec_q():
    use_rec= input('Use reccomended selection volume? yes/no: ')
    medium_q()
    
    if use_rec in ylist:
        cpath['injection radius'] = dd.inj_radius
        cpath['endcap length'] = dd.endcap_len
        cpath['cylinder radius'] = dd.cylRadius
        cpath['cylinder height'] = dd.cylHeight
    elif use_rec in nlist:
        cpath['injection radius'] = input('Injection radius [m]: ')
        cpath['endcap length'] = input('Endcap length [m]: ')
        cpath['cylinder radius'] = input('Cylinder radius [m]: ')
        cpath['cylinder height'] = input('Cylinder height [m]: ')
    else:
        print('invalid input')
        useRec_q()
    
    print('user detector loaded')

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
        global state_key
        state_key = e_type.lower()+'/'
    else:
        print('invalid input')
        event_q()

def interaction_q():
    i_type = input('Which interaction type CC/NC/GR: ')
    if i_type.lower() in ['cc','nc','gr']:
        global state_key
        state_key += i_type.lower()
        cpath['final state 1'] = dd.final_state[state_key][0]
        cpath['final state 2'] = dd.final_state[state_key][1]
    else:
        print('invalid input')
        interaction_q()

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
