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


def config_mims(config, detector) -> None:
    """Set parameters of config so that they are consistent.

    Parameters
    ----------
    config : PrometheusConfig
        Simulation configuration object.
    detector : Detector
        Detector being used for the simulation.
    """
    if detector.medium.name == "WATER":
        config.photon_propagator.name = "olympus"
    elif detector.medium.name == "ICE" and config.photon_propagator.name is None:
        config.photon_propagator.name = "PPC"

    if config.run.random_state_seed is None:
        config.run.random_state_seed = config.run.run_number

    output_prefix = os.path.abspath(
        f"{config.run.storage_prefix}/{config.run.run_number}"
    )
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    if config.run.outfile is None:
        config.run.outfile = f"{output_prefix}_photons.parquet"

    # Find which earth model to use
    base_geofile = os.path.basename(config.detector.geo_file)
    if base_geofile in EARTH_MODEL_DICT:
        earth_model_file = EARTH_MODEL_DICT[base_geofile]
    else:
        earth_model_file = EARTH_MODEL_DICT[detector.medium.name]

    config.detector.offset = [
        detector._offset[0], detector._offset[1], detector._offset[2]
    ]

    injection_config_mims(
        config.injection[config.injection.name],
        detector,
        config.run.nevents,
        config.run.random_state_seed,
        output_prefix,
        earth_model_file,
    )

    lepton_prop_config_mims(
        config.lepton_propagator[config.lepton_propagator.name],
        detector,
        earth_model_file,
    )

    photon_prop_config_mims(config.photon_propagator, output_prefix)
    check_consistency(config)


def check_consistency(config) -> None:
    """Validate the configuration for obvious errors.

    Raises
    ------
    ValueError
        If a numeric range is implausible or a required path does not exist.
    """
    run = config.run
    if run.nevents <= 0:
        raise ValueError(f"run.nevents must be > 0, got {run.nevents}")

    pp_name = config.photon_propagator.name
    if pp_name is None:
        raise ValueError(
            "photon propagator name has not been set. "
            "Is the detector medium recognised?"
        )

    inj_cfg = config.injection[config.injection.name]
    if inj_cfg.inject:
        sim = inj_cfg.simulation
        min_e = sim.minimal_energy
        max_e = sim.maximal_energy
        if min_e is not None and max_e is not None and min_e >= max_e:
            raise ValueError(
                f"injection minimal energy ({min_e}) must be < maximal energy ({max_e})"
            )
        for attr, label in (("diff_xsec", "diff xsec"), ("total_xsec", "total xsec")):
            path = getattr(inj_cfg.paths, attr, None)
            if path is not None and not os.path.exists(path):
                raise ValueError(
                    f"injection paths.{label} does not exist: {path}"
                )

    lp_cfg = config.lepton_propagator[config.lepton_propagator.name]
    tables_path = lp_cfg.paths.tables_path
    if tables_path is not None and not os.path.exists(tables_path):
        raise ValueError(
            f"lepton propagator tables path does not exist: {tables_path}"
        )


def photon_prop_config_mims(config, output_prefix: str) -> None:
    """Fill in computed paths for PPC/PPC_CUDA photon propagator config."""
    name = config.name
    if name not in ("PPC", "PPC_CUDA"):
        return
    ppc_cfg = config[name].paths
    _pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.isabs(ppc_cfg.ppctables):
        ppc_cfg.ppctables = os.path.abspath(
            os.path.join(_pkg_dir, ppc_cfg.ppctables)
        )
    if not os.path.isabs(ppc_cfg.ppc_exe):
        ppc_cfg.ppc_exe = os.path.abspath(
            os.path.join(_pkg_dir, ppc_cfg.ppc_exe)
        )
    if not os.path.isabs(ppc_cfg.ppc_tmpdir):
        ppc_cfg.ppc_tmpdir = os.path.abspath(
            os.path.join(os.path.dirname(output_prefix), ".ppc_tmp")
        )


def lepton_prop_config_mims(config, detector, earth_model_file: str) -> None:
    """Fill in computed fields for lepton propagator config."""
    config.simulation.medium = detector.medium.name.capitalize()
    if config.simulation.propagation_padding is None:
        config.simulation.propagation_padding = detector.outer_radius
        if detector.medium.name == "WATER":
            config.simulation.propagation_padding += 50
        else:
            config.simulation.propagation_padding += 200

    if config.paths.earth_model_location is None:
        config.paths.earth_model_location = (
            f"{RESOURCES_DIR}/earthparams/densities/{earth_model_file}"
        )


def injection_config_mims(
    config,
    detector,
    nevents: int,
    seed: int,
    output_prefix: str,
    earth_model_file: str,
) -> None:
    """Fill in computed fields for the active injector config."""
    if not config.inject:
        return

    if config.paths.earth_model_location is None:
        config.paths.earth_model_location = os.path.abspath(
            f"{RESOURCES_DIR}/earthparams/densities/{earth_model_file}"
        )

    if config.simulation.is_ranged is None:
        config.simulation.is_ranged = False
        if config.simulation.final_state_1 in ("MuMinus", "MuPlus"):
            config.simulation.is_ranged = True

    config.simulation.nevents = nevents
    config.simulation.random_state_seed = seed

    if config.paths.injection_file is None:
        config.paths.injection_file = f"{output_prefix}_LI_output.h5"

    if config.paths.lic_file is None:
        config.paths.lic_file = f"{output_prefix}_LI_config.lic"

    from .geo_utils import get_endcap, get_injection_radius, get_volume
    is_ice = detector.medium.name == "ICE"

    if config.simulation.endcap_length is None:
        config.simulation.endcap_length = get_endcap(detector.module_coords, is_ice)

    if config.simulation.injection_radius is None:
        config.simulation.injection_radius = get_injection_radius(
            detector.module_coords, is_ice
        )

    cyl_radius, cyl_height = get_volume(detector.module_coords, is_ice)
    if config.simulation.cylinder_radius is None:
        config.simulation.cylinder_radius = cyl_radius
    if config.simulation.cylinder_height is None:
        config.simulation.cylinder_height = cyl_height

    int_str = INTERACTION_DICT[(
        config.simulation.final_state_1,
        config.simulation.final_state_2,
    )]

    if int_str in ("CC", "NC"):
        nutype = "nubar"
        if (
            "Bar" in config.simulation.final_state_1
            or "Plus" in config.simulation.final_state_1
        ):
            nutype = "nu"
        if config.paths.diff_xsec is None:
            config.paths.diff_xsec = os.path.abspath(
                f"{config.paths.xsec_dir}/dsdxdy_{nutype}_{int_str}_iso.fits"
            )
        if config.paths.total_xsec is None:
            config.paths.total_xsec = os.path.abspath(
                f"{config.paths.xsec_dir}/sigma_{nutype}_{int_str}_iso.fits"
            )
