from .geo_utils import get_endcap, get_injection_radius, get_volume

def config_mims(config: dict, detector) -> None:
    """Sets parameters of config so that they are consistent
    
    params
    ______
    config: Dictionary specifying the simulation configuration
    detector: Detector being used for the simulation. A lot of 
        the simulation parameters can be set off the geometry of 
        the detector.
    """
    # Set up injection stuff
    injection_config = config["injection"][config["injection"]["name"]]
    if injection_config["inject"]:
        injection_config["simulation"]["random state seed"] = config["general"]["random state seed"]
        is_ice = detector.medium.name == "ICE"
        if injection_config["simulation"]["endcap length"] is None:
            endcap = get_endcap(detector.module_coords, is_ice)
            injection_config["simulation"]["endcap length"] = endcap
        if injection_config["simulation"]["injection radius"] is None:
            inj_radius = get_injection_radius(detector.module_coords, is_ice)
            injection_config["simulation"]["injection radius"] = inj_radius
        cyl_radius, cyl_height = get_volume(detector.module_coords, is_ice)
        if injection_config["simulation"]["cylinder radius"] is None:
            injection_config["simulation"]["cylinder radius"] = cyl_radius
        if injection_config["simulation"]["cylinder height"] is None:
            injection_config["simulation"]["cylinder height"] = cyl_height
    # Set up lepton propagation stuff
    lepton_prop_config = config["lepton propagator"][config["lepton propagator"]["name"]]
    if (
        lepton_prop_config["simulation"]["medium"] is not None and
        lepton_prop_config["simulation"]["medium"].upper()!=detector.medium.name
    ):
        raise ValueError("Detector and lepton propagator have conflicting media")
    lepton_prop_config["simulation"]["medium"] = detector.medium.name.capitalize()
    if lepton_prop_config["simulation"]["inner radius"] is None:
       lepton_prop_config["simulation"]["inner radius"] = (
            detector.outer_radius + lepton_prop_config["simulation"]["propagation padding"]
        )
    return config
