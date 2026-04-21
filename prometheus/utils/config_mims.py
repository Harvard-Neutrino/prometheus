from pathlib import Path

from ..injection.interactions import INTERACTION_DICT

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
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

    output_prefix_path = Path(config.run.storage_prefix) / str(config.run.run_number)
    output_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_prefix_path)
    if config.run.outfile is None:
        config.run.outfile = f"{output_prefix}_photons.parquet"

    # Find which earth model to use
    base_geofile = Path(str(config.detector.geo_file)).name
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
            if path is not None and not Path(path).exists():
                raise ValueError(
                    f"injection paths.{label} does not exist: {path}"
                )

    lp_cfg = config.lepton_propagator[config.lepton_propagator.name]
    tables_path = lp_cfg.paths.tables_path
    if tables_path is not None and not Path(tables_path).exists():
        raise ValueError(
            f"lepton propagator tables path does not exist: {tables_path}"
        )


def photon_prop_config_mims(config, output_prefix: str) -> None:
    """Fill in computed paths for PPC/PPC_CUDA photon propagator config."""
    name = config.name
    if name not in ("PPC", "PPC_CUDA"):
        return
    ppc_cfg = config[name].paths
    _pkg_dir = Path(__file__).resolve().parent.parent
    if not Path(str(ppc_cfg.ppctables)).is_absolute():
        ppc_cfg.ppctables = str((_pkg_dir / ppc_cfg.ppctables).resolve())
    if not Path(str(ppc_cfg.ppc_exe)).is_absolute():
        ppc_cfg.ppc_exe = str((_pkg_dir / ppc_cfg.ppc_exe).resolve())
    if not Path(str(ppc_cfg.ppc_tmpdir)).is_absolute():
        ppc_cfg.ppc_tmpdir = str((Path(output_prefix).parent / ".ppc_tmp").resolve())


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
        config.paths.earth_model_location = str(
            RESOURCES_DIR / "earthparams" / "densities" / earth_model_file
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
        config.paths.earth_model_location = str(
            RESOURCES_DIR / "earthparams" / "densities" / earth_model_file
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
            config.paths.diff_xsec = str(
                Path(str(config.paths.xsec_dir)) / f"dsdxdy_{nutype}_{int_str}_iso.fits"
            )
        if config.paths.total_xsec is None:
            config.paths.total_xsec = str(
                Path(str(config.paths.xsec_dir)) / f"sigma_{nutype}_{int_str}_iso.fits"
            )
