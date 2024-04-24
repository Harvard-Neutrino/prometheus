import numpy as np
import h5py as h5
import awkward as ak
from typing import Iterable

from .. import Particle, PropagatableParticle
from ..injection_event import SIRENInjectionEvent
from ..interactions import Interactions
from .injection import Injection

def apply_detector_offset(
    injection_file: str,
    detector_offset: np.ndarray
) -> None:
    """Translate the injection to a detector-centered coordinate system

    params
    ______
    injection_file: File where the untranslated injection is saved
    detector_offset: Center of the detector in meters
    """
    with h5.File(injection_file, "r+") as h5f:
        injection = h5f[list(h5f.keys())[0]]
        for key in "final_1 final_2 initial".split():
            injection[key]["Position"] = injection[key]["Position"] + detector_offset
        injection["properties"]["x"] = injection["properties"]["x"] + detector_offset[0]
        injection["properties"]["y"] = injection["properties"]["y"] + detector_offset[1]
        injection["properties"]["z"] = injection["properties"]["z"] + detector_offset[2]

class SIRENInjection(Injection):

    def __init__(self, events: Iterable[SIRENInjectionEvent]):
        if not all([isinstance(event, SIRENInjectionEvent) for event in events]):
            raise ValueError("You are trying to make SIREN Injection with non-SIREN events")
        super().__init__(events)
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d["event_weight"] = [x.event_weight for x in self]
        d["secondary_vertex_x"] = [[y for y in x.secondary_vertex_x] for x in self]
        d["secondary_vertex_y"] = [[y for y in x.secondary_vertex_y] for x in self]
        d["secondary_vertex_z"] = [[y for y in x.secondary_vertex_z] for x in self]
        print(d)
        

        return d

def injection_from_SIREN_output(SIREN_file: str) -> SIRENInjection:
    """Creates injection object from a saved SIREN file"""
    with h5.File(SIREN_file, "r") as h5f:
        group = h5f["Events"]
        data = ak.from_buffers(
            ak.forms.from_json(group.attrs["form"]),
            group.attrs["length"],
            {k: np.asarray(v) for k, v in group.items()},
        )
        injection_events = []
        count = 0
        for event in data:
            if -2000001006 not in list(event["secondary_types"][0]): continue
            if event["secondary_momenta"][0][1][0]<50: continue
            if event["secondary_momenta"][1][1][0]<50: continue
            injection_events.append(injection_event_from_SIREN(event))
            count += 1
            if count >=10: break
        return SIRENInjection(injection_events)

INTERACTION_CONVERTER = {
    (12, -2000001006, 11): Interactions.CHARGED_CURRENT,
    (14, -2000001006, 13): Interactions.CHARGED_CURRENT,
    (16, -2000001006, 15): Interactions.CHARGED_CURRENT,
    (12, 11, -2000001006): Interactions.CHARGED_CURRENT,
    (14, 13, -2000001006): Interactions.CHARGED_CURRENT,
    (16, 15, -2000001006): Interactions.CHARGED_CURRENT,
    (-12, -2000001006, -11): Interactions.CHARGED_CURRENT,
    (-14, -2000001006, -13): Interactions.CHARGED_CURRENT,
    (-16, -2000001006, -15): Interactions.CHARGED_CURRENT,
    (-12, -11, -2000001006): Interactions.CHARGED_CURRENT,
    (-14, -13, -2000001006): Interactions.CHARGED_CURRENT,
    (-16, -15, -2000001006): Interactions.CHARGED_CURRENT,
    (12, 12, -2000001006): Interactions.NEUTRAL_CURRENT,
    (14, 14, -2000001006): Interactions.NEUTRAL_CURRENT,
    (16, 16, -2000001006): Interactions.NEUTRAL_CURRENT,
    (12, -2000001006, 12): Interactions.NEUTRAL_CURRENT,
    (14, -2000001006, 14): Interactions.NEUTRAL_CURRENT,
    (16, -2000001006, 16): Interactions.NEUTRAL_CURRENT,
    (-12, -12,-2000001006): Interactions.NEUTRAL_CURRENT,
    (-14, -14,-2000001006): Interactions.NEUTRAL_CURRENT,
    (-16, -16,-2000001006): Interactions.NEUTRAL_CURRENT,
    (-12,-2000001006, -12): Interactions.NEUTRAL_CURRENT,
    (-14,-2000001006, -14): Interactions.NEUTRAL_CURRENT,
    (-16,-2000001006, -16): Interactions.NEUTRAL_CURRENT,
    (-12, -2000001006, -2000001006): Interactions.GLASHOW_RESONANCE,
    (-12,-12, 11): Interactions.GLASHOW_RESONANCE,
    (-12,-14, 13): Interactions.GLASHOW_RESONANCE,
    (-12,-16, 15): Interactions.GLASHOW_RESONANCE,
    (-12, 11,-12): Interactions.GLASHOW_RESONANCE,
    (-12, 13,-14): Interactions.GLASHOW_RESONANCE,
    (-12, 15,-16): Interactions.GLASHOW_RESONANCE,
    (14, 13, -13): Interactions.DIMUON,
    (14, -13, 13): Interactions.DIMUON,
    (-14, -13, 13): Interactions.DIMUON,
    (-14, 13, -13): Interactions.DIMUON,
}

for target_fs in [2212,1000080160,-2000001006]:
    INTERACTION_CONVERTER[(12, target_fs, 5914)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(12, 5914, target_fs)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(14, target_fs, 5914)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(14, 5914, target_fs)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(16, target_fs, 5914)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(16, 5914, target_fs)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(12, target_fs, 5914)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(12, 5914, target_fs)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(14, target_fs, 5914)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(14, 5914, target_fs)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(16, target_fs, 5914)] = Interactions.DIPOLE
    INTERACTION_CONVERTER[(16, 5914, target_fs)] = Interactions.DIPOLE

def propagate(particle_id: int) -> bool:
    """
    determine which p
    """
    return particle_id in [
        -2000001006,2212, # hadrons
        11,13,15, # lep+
        -11,-13,-15, #lep-
        12,14,16, # nu
        -12,-14,-16, # antinu
        111,211,311, # pions
        22, # gamma
        ]

def injection_event_from_SIREN(injection: ak.highlevel.Record,
                               detector_offset = np.array([0,0,6372184-6374134])) -> SIRENInjectionEvent:
    """Create an injection event from SIREN h5 group and index

    params
    ______
    injection: awkward record for the event

    returns
    _______
    event: Prometheus SIRENInjectionEvent corresponding to input
    """
    event_weight = injection["event_weight"]
    final_states = []
    for i_int in range(injection["num_interactions"]):

        vertex =  + injection["vertex"][i_int] + detector_offset

        primary_momentum = injection["primary_momentum"][i_int]
        density_variables = [] # todo: update SIREN serialization to include this
        secondary_vertex_x = []
        secondary_vertex_y = []
        secondary_vertex_z = []
        if(injection["parent_idx"][i_int]==-1):
            # primary interaction
            time = 0
            initial_state = Particle(
                injection["primary_type"][i_int],
                primary_momentum[0],
                vertex,
                primary_momentum[1:]/np.linalg.norm(primary_momentum[1:]),
                time,
                None
            )
            vertex_x = vertex
            vertex_y = vertex
            vertex_z = vertex
        else:
            parent_vertex = injection["vertex"][injection["parent_idx"][i_int]]
            distance = np.linalg.norm(vertex - parent_vertex)
            beta = np.linalg.norm(primary_momentum[1:])/primary_momentum[0]
            velocity = 3.e-1 * beta # m/ns
            time = distance/velocity
        for final_ctr in range(injection["num_secondaries"][i_int]):
            if not propagate(injection["secondary_types"][i_int][final_ctr]): continue
            secondary_momentum = injection["secondary_momenta"][i_int][final_ctr]
            if injection["secondary_types"][i_int][final_ctr]==22:
                # photon case, approximate pair production
                final_state1 = PropagatableParticle(
                    11,
                    secondary_momentum[0]/2,
                    vertex,
                    secondary_momentum[1:]/np.linalg.norm(secondary_momentum[1:]),
                    time,
                    None,
                    initial_state
                )
                final_state2 = PropagatableParticle(
                    -11,
                    secondary_momentum[0]/2,
                    vertex,
                    secondary_momentum[1:]/np.linalg.norm(secondary_momentum[1:]),
                    time,
                    None,
                    initial_state
                )
                final_states.append(final_state1)
                final_states.append(final_state2)
                secondary_vertex_x.append(vertex[0])
                secondary_vertex_y.append(vertex[1])
                secondary_vertex_z.append(vertex[2])
                secondary_vertex_x.append(vertex[0])
                secondary_vertex_y.append(vertex[1])
                secondary_vertex_z.append(vertex[2])
            else:
                final_state = PropagatableParticle(
                    injection["secondary_types"][i_int][final_ctr],
                    secondary_momentum[0],
                    vertex,
                    secondary_momentum[1:]/np.linalg.norm(secondary_momentum[1:]),
                    time,
                    None,
                    initial_state
                )
                final_states.append(final_state)
                secondary_vertex_x.append(vertex[0])
                secondary_vertex_y.append(vertex[1])
                secondary_vertex_z.append(vertex[2])
        if(injection["parent_idx"][i_int]==-1):
            interaction = INTERACTION_CONVERTER[(
                initial_state.pdg_code,
                *injection["secondary_types"][i_int]
            )]
        
        
        
    event = SIRENInjectionEvent(
        initial_state,
        final_states,
        interaction,
        vertex_x,
        vertex_y,
        vertex_z,
        density_variables,
        event_weight,
        secondary_vertex_x,
        secondary_vertex_y,
        secondary_vertex_z,
    )
    return event
