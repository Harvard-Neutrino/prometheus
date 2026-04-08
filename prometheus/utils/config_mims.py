import os

from ..injection.interactions import INTERACTION_DICT

RESOURCES_DIR = os.path.abspath(f"{os.path.dirname(__file__)}/../../resources/")
EARTH_MODEL_DICT = {
    "gvd.geo": "PREM_gvd.dat",
    "icecube.geo": "PREM_south_pole.dat",
    "icecube_gen2.geo": "PREM_south_pole.dat",
    "icecube_upgrade.geo": "PREM_south_pole.dat",
    "orca.geo": "PREM_orca.dat",
    "arca.geo": "PREM_arca.dat",
    "pone.geo": "PREM_pone.dat",
    # The following options are used in case another file is provided
    "WATER": "PREM_water.dat",
    "ICE": "PREM_south_pole.dat",
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

    if detector.medium.name=="WATER":
        config["photon propagator"]["name"] = "olympus"
    elif detector.medium.name=="ICE" and config["photon propagator"]["name"] is None:
        config["photon propagator"]["name"] = "PPC"

    run_config = config["run"]
    if run_config["random state seed"] is None:
        run_config["random state seed"] = run_config["run number"]

    output_prefix = os.path.abspath(f"{config['run']['storage prefix']}/{config['run']['run number']}")
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    if config["run"]["outfile"] is None:
        config["run"]["outfile"] = (
            f"{output_prefix}_photons.parquet"
        )

    # Find which earth model we think we should be using
    earth_model_file = None
    base_geofile = os.path.basename(config["detector"]["geo file"])
    if base_geofile in EARTH_MODEL_DICT.keys():
        earth_model_file = EARTH_MODEL_DICT[base_geofile]
    else:
        earth_model_file = EARTH_MODEL_DICT[detector.medium.name]

    config["detector"]["offset"] = [detector._offset[0], detector._offset[1], detector._offset[2]]

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
        config["photon propagator"],
        output_prefix
    )
    check_consistency(config)

def check_consistency(config: dict) -> None:
    """Validate the configuration for obvious errors.

    Raises
    ------
    KeyError
        If a required top-level section is missing.
    ValueError
        If a numeric range is implausible or a required path does not exist.
    """
    # Required top-level sections
    for section in ("run", "detector", "injection", "lepton propagator", "photon propagator"):
        if section not in config:
            raise KeyError(f"Config is missing required section: '{section}'")

    run = config["run"]
    if run.get("nevents", 0) <= 0:
        raise ValueError(f"run.nevents must be > 0, got {run.get('nevents')}")
    if run.get("random state seed") is not None and not isinstance(run["random state seed"], int):
        raise ValueError("run.random state seed must be an integer or None")

    # Photon propagator name must have been resolved by now
    pp_name = config["photon propagator"].get("name")
    if pp_name is None:
        raise ValueError(
            "photon propagator name has not been set. "
            "Is the detector medium recognised?"
        )

    # Injection: if LeptonInjector is active and inject=True, cross-section files must exist
    inj_name = config["injection"].get("name", "")
    if inj_name in config["injection"]:
        inj_cfg = config["injection"][inj_name]
        if inj_cfg.get("inject", False):
            sim = inj_cfg.get("simulation", {})
            min_e = sim.get("minimal energy")
            max_e = sim.get("maximal energy")
            if min_e is not None and max_e is not None and min_e >= max_e:
                raise ValueError(
                    f"injection minimal energy ({min_e}) must be < maximal energy ({max_e})"
                )
            for path_key in ("diff xsec", "total xsec"):
                path = inj_cfg.get("paths", {}).get(path_key)
                if path is not None and not os.path.exists(path):
                    raise ValueError(
                        f"injection paths.{path_key} does not exist: {path}"
                    )

    # Lepton propagator tables path should exist if set
    lp_name = config["lepton propagator"].get("name", "")
    if lp_name in config["lepton propagator"]:
        tables_path = (
            config["lepton propagator"][lp_name]
            .get("paths", {})
            .get("tables path")
        )
        if tables_path is not None and not os.path.exists(tables_path):
            raise ValueError(
                f"lepton propagator tables path does not exist: {tables_path}"
            )

def photon_prop_config_mims(config: dict, output_prefix: str) -> None:
    name = config.get("name")
    if name not in ("PPC", "PPC_CUDA"):
        return
    ppc_cfg = config[name]["paths"]
    # Relative paths in the config are expressed relative to the prometheus
    # package directory (prometheus/prometheus/). Resolve them to absolute paths
    # so PPC works regardless of the current working directory.
    _pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.isabs(ppc_cfg["ppctables"]):
        ppc_cfg["ppctables"] = os.path.abspath(
            os.path.join(_pkg_dir, ppc_cfg["ppctables"])
        )
    if not os.path.isabs(ppc_cfg["ppc_exe"]):
        ppc_cfg["ppc_exe"] = os.path.abspath(
            os.path.join(_pkg_dir, ppc_cfg["ppc_exe"])
        )
    # Resolve ppc_tmpdir to an absolute path derived from output_prefix
    if not os.path.isabs(ppc_cfg["ppc_tmpdir"]):
        ppc_cfg["ppc_tmpdir"] = os.path.abspath(
            os.path.join(os.path.dirname(output_prefix), ".ppc_tmp")
        )


def lepton_prop_config_mims(config: dict, detector, earth_model_file: str) -> None:
    config["simulation"]["medium"] = detector.medium.name.capitalize()
    if config["simulation"]["propagation padding"] is None:
        config["simulation"]["propagation padding"] = detector.outer_radius
        if detector.medium.name=="WATER":
            config["simulation"]["propagation padding"] += 50
        else:
            config["simulation"]["propagation padding"] += 200

    if config["paths"]["earth model location"] is None:
#        if earth_model_file is None:
#            earth_model_file = EARTH_MODEL_DICT[detector.medium.name]
        config["paths"]["earth model location"] = (
            f"{RESOURCES_DIR}/earthparams/densities/{earth_model_file}"
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
        # Work on a shallow copy so the caller's dict is not permanently mutated.
        config = dict(config)
        config.pop("simulation", None)
        return

    if config["paths"]["earth model location"] is None:
        #earth_model_file = EARTH_MODEL_DICT[detector.medium.name]
        config["paths"]["earth model location"] = (
            os.path.abspath(f"{RESOURCES_DIR}/earthparams/densities/{earth_model_file}")
        )

    if config["simulation"]["is ranged"] is None:
        config["simulation"]["is ranged"] = False
        if config["simulation"]["final state 1"] in "MuMinus MuPlus".split():
            config["simulation"]["is ranged"] = True

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
