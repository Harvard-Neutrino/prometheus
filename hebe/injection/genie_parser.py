# genie_parser.py
# Authors: Stephan Meighen-Berger
# Utility functions to load and parse GENIE data

import pandas as pd
import numpy as np
import uproot

def genie_parser(events) -> pd.DataFrame:
    """ function to fetch the relevant information from genie events (in rootracker format)

    Parameters
    ----------
    events : dict
        The genie events

    Returns
    -------
    pd.DataFrame
        Data frame containing the relevant information
    """
    dic = {}
    dic['event_description'] = events['EvtCode/fString'].array(library='np')  # String describing the event
    dic['event_id'] = events['EvtNum'].array(library="np")  # Number of the event
    dic['event_children'] = events['StdHepN'].array(library="np")  # NUmber of the generated sub-particles
    dic['event_prob'] = events['EvtProb'].array(library="np")  # Probability of the event
    dic['event_xsec'] = events['EvtXSec'].array(library="np")  # Total xsec of the event
    dic['event_pdg_id'] = events['StdHepPdg'].array(library="np")  # PDG ids of all produced particles
    dic['event_momenta'] = events['StdHepP4'].array(library="np")  # Momenta of the particles
    tmp = events['EvtVtx'].array(library="np")  # Position of the particles
    dic['event_vertex'] = [np.array(vtx) for vtx in tmp]
    dic['event_coords'] = events['StdHepX4'].array(library="np")  # Position of the particles
    dic['event_weight'] = events['EvtWght'].array(library='np')  # Weight of the events
    tmp = events['StdHepStatus'].array(library="np")  # Status of the particle
    # Converting the codes
    particle_dic = {
        0: 'initial',
        1: 'final',
        2: 'intermediate',
        3: 'decayed',
        11: 'nucleon target',
        12: 'DIS pre-frag hadronic state',
        13: 'resonance',
        14: 'hadron in nucleus',
        15: 'final nuclear',
        16: 'nucleon cluster target',
    }
    new_arr = np.array([[
        particle_dic[particle] for particle in event
    ] for event in tmp], dtype=object)
    dic['event_status'] = new_arr
    return pd.DataFrame.from_dict(dic)

def final_parser(parsed_events: pd.DataFrame) -> pd.DataFrame:
    """ fetches the final states

    Parameters
    ----------
    parsed_events : pd.DataFrame
        The parsed events

    Returns
    -------
    pd.DataFrame
        The inital + final state info
    """
    inital_energies_inj = np.array([event[0][3] for event in parsed_events['event_momenta']])
    inital_momenta_inj = [np.array(event[0][:3]) for event in parsed_events['event_momenta']]
    inital_energies_target = np.array([event[1][3] for event in parsed_events['event_momenta']])
    inital_id_inj = np.array([event[0] for event in parsed_events['event_pdg_id']])
    inital_id_target = np.array([event[1] for event in parsed_events['event_pdg_id']])
    final_ids = np.array([np.where(event == np.array('final'), True, False) for event in parsed_events['event_status']], dtype=object)
    children_ids = np.array([
        event[final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_pdg_id'])
    ], dtype=object)
    children_energy = np.array([
        event[:, 3][final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_momenta'])
    ], dtype=object)
    children_momenta = np.array([
        event[:, :3][final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_momenta'])
    ], dtype=object)
    final_ids = np.array([np.where(event == np.array('final nuclear'), True, False) for event in parsed_events['event_status']], dtype=object)
    children_nuc_ids = np.array([
        event[final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_pdg_id'])
    ], dtype=object)
    children_nuc_energy = np.array([
        event[:, 3][final_ids[id_event]] for id_event, event in enumerate(parsed_events['event_momenta'])
    ], dtype=object)
    dic = {}
    dic['event_descr'] = parsed_events['event_description']
    dic['event_xsec'] = parsed_events['event_xsec']
    dic['event_vertex'] = parsed_events.event_vertex
    dic['init_inj_e'] = inital_energies_inj
    dic['init_inj_p'] = inital_momenta_inj
    dic['init_target_e'] = inital_energies_target
    dic['init_inj_id'] = inital_id_inj
    dic['init_target_id'] = inital_id_target
    dic['final_ids'] = children_ids
    dic['final_e'] = children_energy
    dic['final_p'] = children_momenta
    dic['final_nuc_ids'] = children_nuc_ids
    dic['final_nuc_e'] = children_nuc_energy
    return pd.DataFrame.from_dict(dic)

def genie_loader(filepath: str) -> pd.DataFrame:
    """ Loads and parses GENIE data

    Parameters
    ----------
    filepath: str
        Path to the GENIE file

    Returns
    -------
    pd.DataFrame
        The parsed genie data
    """
    with uproot.open(filepath) as file:
        return final_parser(genie_parser(file['gRooTracker']))