import h5py as h5
import numpy as np

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

def make_new_LI_injection(
    path_dict: dict,
    injection_specs: dict,
    detector_offset: np.ndarray
) -> None:
    """Make a new injection with LeptonInjector

    params
    ______
    path_dict: dictionary specifying all the necessary pathing information
    injection_specs: dictionary specifying all the injection configuration
        settings
    detector_offset: Center of the detector in meters

    """
    import os
    print('Importing LeptonInjector')
    try:
        try:
            print('Trying default pythonpath')
            import EarthModelService as em
            import LeptonInjector as LI
        except ImportError:
            import sys
            print('Trying custom path set in config')
            print(f"The path is {path_dict['install location']}")
            sys.path.append(path_dict['install location'])
            import EarthModelService as em
            import LeptonInjector as LI
    except ImportError:
        raise ImportError("LeptonInjector not found!")
    n_events = injection_specs["nevents"]
    #xs_folder = os.path.join(
    #    os.path.dirname(__file__),
    #    path_dict["xsec dir"]
    #)
    diff_xs = path_dict['diff xsec']
    total_xs = path_dict['total xsec']
    is_ranged = injection_specs["is ranged"]
    particles = []
    for names in [injection_specs["final state 1"], injection_specs["final state 2"]]:
    #for id_name, names in enumerate([
    #    injection_specs["final state 1"],
    #    injection_specs["final state 2"]
    #]):
        particles.append(getattr(LI.Particle.ParticleType, names))
    
    the_injector = LI.Injector(
        injection_specs["nevents"],
        particles[0],
        particles[1],
        diff_xs,
        total_xs,
        is_ranged
    )
    min_E = injection_specs["minimal energy"]
    max_E = injection_specs["maximal energy"]
    gamma = injection_specs["power law"]
    min_zenith = np.radians(injection_specs["min zenith"])
    max_zenith = np.radians(injection_specs["max zenith"])
    min_azimuth = np.radians(injection_specs["min azimuth"])
    max_azimuth = np.radians(injection_specs["max azimuth"])
    inject_radius = injection_specs["injection radius"]
    endcap_length = injection_specs["endcap length"]
    cyinder_radius = injection_specs["cylinder radius"]
    cyinder_height = injection_specs["cylinder height"]
    # construct the controller
    controller = LI.Controller(
        the_injector, min_E, max_E, gamma, min_azimuth,
        max_azimuth, min_zenith, max_zenith, 
        inject_radius, endcap_length, cyinder_radius, cyinder_height
    )
    earth_model_dir = "/".join(path_dict["earth model location"].split("/")[:-2]) + "/"
    earth_model_name = path_dict["earth model location"].split("/")[-1].split(".")[0]
    earth = em.EarthModelService(
        "Zorg",
        earth_model_dir,
        [earth_model_name],
        ["Standard"],
        "NoIce",
        0.0,
        -detector_offset[2]
    )
    controller.SetEarthModelService(earth)
    controller.setSeed(injection_specs["random state seed"])
    controller.NameOutfile(path_dict["injection file"])
    controller.NameLicFile(path_dict["lic file"])

    # run the simulation
    controller.Execute()
    # Translate injection to detector coordinate system
    apply_detector_offset(path_dict["injection file"], detector_offset)
