from .geo_utils import get_endcap, get_injection_radius, get_volume

INTERACTION_DICT = {
    ("EMinus", "Hadrons"): "CC",
    ("MuMinus", "Hadrons"): "CC",
    ("TauMinus", "Hadrons"): "CC",
    ("NuEMinus", "Hadrons"): "NC",
    ("NuMuMinus", "Hadrons"): "NC",
    ("NuTauMinus", "Hadrons"): "NC",
}

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
    run_config = config["run"]
    if run_config["random state seed"] is None:
        run_config["random state seed"] = run_config["run number"]
    injection_config = config["injection"][config["injection"]["name"]]
    if injection_config["inject"]:
        injection_config["simulation"]["nevents"] = config["run"]["nevents"]
        injection_config["simulation"]["random state seed"] = config["run"]["random state seed"]
        if injection_config["paths"]["injection file"] is None:
            injection_config["paths"]["injection file"] = (
                f"{config['run']['storage prefix']}/{config['run']['run number']}_LI_output.h5"
            )
        if injection_config["paths"]["lic file"] is None:
            injection_config["paths"]["lic file"] = (
                f"{config['run']['storage prefix']}/{config['run']['run number']}_LI_config.lic"
            )
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
        int_str = INTERACTION_DICT[(
            injection_config["simulation"]["final state 1"],
            injection_config["simulation"]["final state 2"]
        )]
        if (
            "Bar" in injection_config["simulation"]["final state 1"] or \
            "Plus" in injection_config["simulation"]["final state 1"]
        ):
            nutype = "nu"
        else:
            nutype = "nubar"
        if injection_config["paths"]["diff xsec"] is None:
            injection_config["paths"]["diff xsec"] = (
                f"{injection_config['paths']['xsec dir']}/dsdxdy_{nutype}_{int_str}_iso.fits"
            )
        if injection_config["paths"]["total xsec"] is None:
            injection_config["paths"]["total xsec"] = (
                f"{injection_config['paths']['xsec dir']}/sigma_{nutype}_{int_str}_iso.fits"
            )
        #if injection_config["paths"]["injection file"] is None:
        #    injection_config["paths"]["output name"] = (
        #        f"{config['run']['storage location']}/{config['run']['random state seed']}_LI_output.h5"
        #    )
        #if injection_config["paths"]["lic name"] is None:
        #    injection_config["paths"]["lic name"] = (
        #        f"{config['run']['storage location']}/{config['run']['random state seed']}_LI_config.lic"
        #    )
    # Set up lepton propagation stuff
    lepton_prop_config = config["lepton propagator"][config["lepton propagator"]["name"]]
    if (
        lepton_prop_config["simulation"]["medium"] is not None and
        lepton_prop_config["simulation"]["medium"].upper()!=detector.medium.name
    ):
        raise ValueError("Detector and lepton propagator have conflicting media")
    lepton_prop_config["simulation"]["medium"] = detector.medium.name.capitalize()
    if lepton_prop_config["paths"]["earth model location"] is None:
        lepton_prop_config["paths"]["earth model location"] = injection_config["paths"]["earth model location"]
    if lepton_prop_config["simulation"]["inner radius"] is None:
       lepton_prop_config["simulation"]["inner radius"] = (
            detector.outer_radius + lepton_prop_config["simulation"]["propagation padding"]
        )
    pp_config = config["photon propagator"][config["photon propagator"]["name"]]
    if pp_config["paths"]["outfile"] is None:
        pp_config["paths"]["outfile"] = (
            f"{config['run']['storage prefix']}/{config['run']['run number']}_photons.parquet"
        )
