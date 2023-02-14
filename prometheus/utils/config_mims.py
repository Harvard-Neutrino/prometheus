import os

RESOURCES_DIR = f"{os.path.dirname(__file__)}/../../resources/"
INTERACTION_DICT = {
    ("EMinus", "Hadrons"): "CC",
    ("MuMinus", "Hadrons"): "CC",
    ("TauMinus", "Hadrons"): "CC",
    ("NuEMinus", "Hadrons"): "NC",
    ("NuMuMinus", "Hadrons"): "NC",
    ("NuTauMinus", "Hadrons"): "NC",
}
EARTH_MODEL_DICT = {
    "gvd.geo": "PREM_gvd.dat",
    "icecube.geo": "PREM_mmc.dat",
    "icecube_gen2.geo": "PREM_mmc.dat",
    "icecube_upgrade.geo": "PREM_mmc.dat",
    "orca.geo": "PREM_orca.dat",
    "arca.geo": "PREM_arca.dat",
    "pone.geo": "PREM_pone.dat",
    # The following options are used in case another file is provided
    "WATER": "PREM_water.dat",
    "ICE": "PREM_mmc.dat",
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
    output_prefix = os.path.abspath(f"{config['run']['storage prefix']}/{config['run']['run number']}")

    # Find which earth model we think we should be using
    earth_model_file = None
    base_geofile = os.path.basename(config["detector"]["geo file"])
    if base_geofile in EARTH_MODEL_DICT.keys():
        earth_model_file = EARTH_MODEL_DICT[base_geofile]

    injection_config_mims(
        config["injection"][config["injection"]["name"]],
        detector,
        config["run"]["nevents"],
        config["run"]["random state seed"],
        output_prefix,
        earth_model_file
    )

    lepton_prop_config_mims(
        config["lepton propagator"][config["lepton propagator"]["name"]],
        detector,
        earth_model_file
    )

    photon_prop_config_mims(
        config["photon propagator"][config["photon propagator"]["name"]],
        output_prefix
    )
    check_consistency(config)

def check_consistency(config: dict) -> None:
    # TODO check whether medium is knowable
    # TODO check if medium is consistent
    
    
    pass
    #if (
    #    config["simulation"]["medium"] is not None and
    #    config["simulation"]["medium"].upper()!=detector.medium.name
    #):
    #    raise ValueError("Detector and lepton propagator have conflicting media")

def photon_prop_config_mims(config: dict, output_prefix: str) -> None:
    if config["paths"]["outfile"] is None:
        config["paths"]["outfile"] = (
            f"{output_prefix}_photons.parquet"
        )


def lepton_prop_config_mims(config: dict, detector, earth_model_file: str) -> None:
    config["simulation"]["medium"] = detector.medium.name.capitalize()
    if config["paths"]["earth model location"] is None:
        # TODO what are you doing here, Jeff ?
        config["paths"]["earth model location"] = config["paths"]["earth model location"]
    if config["simulation"]["inner radius"] is None:
       config["simulation"]["inner radius"] = (
            detector.outer_radius + config["simulation"]["propagation padding"]
        )
    if config["paths"]["earth model location"] is None:
        if earth_model_file is None:
            earth_model_file = EARTH_MODEL_DICT[detector.medium.name]
        config["paths"]["earth model location"] = (
            os.path.abspath(f"{RESOURCES_DIR}/earthparams/densities/{earth_model_file}")
        )

def injection_config_mims(
    config:dict,
    detector,
    nevents: int,
    seed: int,
    output_prefix: str,
    earth_model_file: str
) -> None:

    if not config["inject"]:
        del config["simulation"]
        return

    if config["paths"]["earth model location"] is None:
        if earth_model_file is None:
            earth_model_file = EARTH_MODEL_DICT[detector.medium.name]
        config["paths"]["earth model location"] = (
            os.path.abspath(f"{RESOURCES_DIR}/earthparams/densities/{earth_model_file}")
        )

    config["simulation"]["nevents"] = nevents
    # Make sure seeding is consistent
    config["simulation"]["random state seed"] = seed

    # Name the h5 file
    if config["paths"]["injection file"] is None:
        config["paths"]["injection file"] = (
            f"{output_prefix}_LI_output.h5"
        )
    # Name the lic file
    if config["paths"]["lic file"] is None:
        config["paths"]["lic file"] = (
            f"{output_prefix}_LI_config.lic"
        )

    from .geo_utils import get_endcap, get_injection_radius, get_volume
    # TODO we shouldn't set the scattering length like this
    is_ice = detector.medium.name == "ICE"
    # Set the endcap length
    if config["simulation"]["endcap length"] is None:
        endcap = get_endcap(detector.module_coords, is_ice)
        config["simulation"]["endcap length"] = endcap
    # Set the injection radius
    if config["simulation"]["injection radius"] is None:
        inj_radius = get_injection_radius(detector.module_coords, is_ice)
        config["simulation"]["injection radius"] = inj_radius
    # Set the cylinder radius and height
    cyl_radius, cyl_height = get_volume(detector.module_coords, is_ice)
    if config["simulation"]["cylinder radius"] is None:
        config["simulation"]["cylinder radius"] = cyl_radius
    if config["simulation"]["cylinder height"] is None:
        config["simulation"]["cylinder height"] = cyl_height

    # Set the interaction
    int_str = INTERACTION_DICT[(
        config["simulation"]["final state 1"],
        config["simulation"]["final state 2"]
    )]

    
    if int_str in "CC NC".split():
        # Set cross section spline paths
        nutype = "nubar"
        if (
            "Bar" in config["simulation"]["final state 1"] or \
            "Plus" in config["simulation"]["final state 1"]
        ):
            nutype = "nu"
        if config["paths"]["diff xsec"] is None:
            config["paths"]["diff xsec"] = (
                os.path.abspath(f"{config['paths']['xsec dir']}/dsdxdy_{nutype}_{int_str}_iso.fits")
            )
        if config["paths"]["total xsec"] is None:
            config["paths"]["total xsec"] = (
                os.path.abspath(f"{config['paths']['xsec dir']}/sigma_{nutype}_{int_str}_iso.fits")
            )
    else:
        # Glashow resonance xs is not set by splines
        del config["paths"]["xsec dir"]
        del config["paths"]["diff xsec"] 
        del config["paths"]["total xsec"] 
