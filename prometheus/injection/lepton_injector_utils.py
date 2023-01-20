import h5py as h5
import numpy as np

from .injection import Injection
from .interactions import Interactions

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

def injection_event_from_LI(injection, idx):
    initial_state= Particle(
        injection["properties"]["initialType"][idx],
        injection["initial"]["Energy"][idx],
        injection["initial"]["Position"][idx],
        np.array([
            np.sin(injection["initial"]["Direction"][idx][0]) * np.cos(injection["initial"]["Direction"][idx][1]),
            np.sin(injection["initial"]["Direction"][idx][0]) * np.sin(injection["initial"]["Direction"][idx][1]),
            np.cos(injection["initial"]["Direction"][idx][0]),
        ])
    )
    final_states = []
    for final_ctr in [1,2]:
        final_state = PropagatableParticle(
            injection["properties"][f"finalType{final_ctr}"][idx],
            injection[f"final_{final_ctr}"]["Energy"][idx],
            injection[f"final_{final_ctr}"]["Position"][idx],
            np.array([
                np.sin(injection[f"final_{final_ctr}"]["Direction"][idx][0]) * np.cos(injection[f"final_{final_ctr}"]["Direction"][idx][1]),
                np.sin(injection[f"final_{final_ctr}"]["Direction"][idx][0]) * np.sin(injection[f"final_{final_ctr}"]["Direction"][idx][1]),
                np.cos(injection[f"final_{final_ctr}"]["Direction"][idx][0]),
            ])
        )
        final_states.append(final_state)
    interaction = INTERACTION_CONVERTER[(
        initial_state.pdg_code,
        final_states[0].pdg_code,
        final_states[1].pdg_code,
    )]
    return InjectionEvent(initial_state, final_states, interaction)

def injection_from_LI_output(LI_file:str) -> Injection:
    """Creates injection objection from a saved LI file"""
    with h5.File(LI_file, "r") as h5f:
        injectors = list(h5f.keys())
        if len(injectors) > 1:
            raise ValueError("Too many injectors")
        injection = h5f[injectors[0]]
        injection_events = []
        for idx in range(injection["initial"].shape[0]):
            injection_events.append(
                injection_event_from_LI(injection, idx)
            )
        return Injection(injection_events)

def apply_detector_offset(
    injection_file: str,
    detector_offset: np.ndarray
) -> None:
    """translates the injection to a detector-centered coordinate system

    params
    ______
    injection_file: File where the untranslated injection is saved
    detector_offset: Center of the detector
    """
    with h5.File(injection_file, "r+") as h5f:
        injection = h5f[list(h5f.keys())[0]]
        for key in "final_1 final_2 initial".split():
            injection[key]["Position"] = injection[key]["Position"] + detector_offset
        injection["properties"]["x"] = injection["properties"]["x"] + detector_offset[0]
        injection["properties"]["y"] = injection["properties"]["y"] + detector_offset[1]
        injection["properties"]["z"] = injection["properties"]["z"] + detector_offset[2]

def make_new_injection(
    path_dict: dict,
    injection_specs: dict,
    detector_offset: np.ndarray
) -> None:
    """Creates a new lepton injector simulation

    params
    ______
    path_dict: dictionary specifying relevant paths for the simulation
    injection_specs: dictionary specifying the injection parameters like
        energy, direction, etc.
    detector_offset: the physical position of the center of the detector. The
        injection is always centered around the origin, so we need to translate
        it to the center of the detector
    """
    import os
    try:
        import LeptonInjector as LI
    except ImportError:
        raise ImportError("LeptonInjector not found!")
    print("Setting up the LI")
    print("Fetching parameters and setting paths")
    xs_folder = os.path.join(
        os.path.dirname(__file__),
        path_dict["xsec location"]
    )
    print("Setting the simulation parameters for the LI")
    n_events = injection_specs["nevents"]
    diff_xs = f"{path_dict['xsec location']}/{path_dict['diff xsec']}"
    total_xs = f"{path_dict['xsec location']}/{path_dict['total xsec']}"
    is_ranged = injection_specs["is ranged"]
    particles = []
    for id_name, names in enumerate([
        injection_specs["final state 1"],
        injection_specs["final state 2"]
    ]):
        particles.append(getattr(LI.Particle.ParticleType, names))
    
    print("Setting up the LI object")
    the_injector = LI.Injector(
        injection_specs["nevents"],
        particles[0],
        particles[1],
        diff_xs,
        total_xs,
        is_ranged
    )
    print("Setting injection parameters")

    # define some defaults
    min_E = injection_specs["minimal energy"]
    max_E = injection_specs["maximal energy"]
    gamma = injection_specs["power law"]
    min_zenith = np.radians(injection_specs["min zenith"])
    max_zenith = np.radians(injection_specs["max zenith"])
    min_azimuth = np.radians(injection_specs["min azimuth"])
    max_azimuth = np.radians(injection_specs["max azimuth"])
    inject_radisu = injection_specs["injection radius"]
    endcap_length = injection_specs["endcap length"]
    cyinder_radius = injection_specs["cylinder radius"]
    cyinder_height = injection_specs["cylinder height"]
    print("Building the injection handler")
    # construct the controller
    if is_ranged:
        controller = LI.Controller(
            the_injector, min_E, max_E, gamma, min_azimuth,
            max_azimuth, min_zenith, max_zenith, 
        )
    else:
        controller = LI.Controller(
            the_injector, min_E, max_E, gamma, min_azimuth,
            max_azimuth, min_zenith, max_zenith,
            inject_radius, endcap_length, cyinder_radius, cyinder_height
        )
    print("Defining the earth model")
    path_to = os.path.join(
        os.path.dirname(__file__),
        xs_folder, path_dict["earth model location"]
    )
    print(
        f"Earth model location to use: {path_to} with the model {injection_specs['earth model']}"
    )
    print(injection_specs["earth model"], path_dict["earth model location"])
    controller.SetEarthModel(injection_specs["earth model"], path_to)
    print("Setting the seed")
    controller.setSeed(injection_specs["random state seed"])
    print("Defining the output location")
    controller.NameOutfile(path_dict["output name"])
    controller.NameLicFile(path_dict["lic name"])

    # run the simulation
    controller.Execute()
    # Translate injection to detector coordinate system
    apply_detector_offset(path_dict["output name"], detector_offset)
