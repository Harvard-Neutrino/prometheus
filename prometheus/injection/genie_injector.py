import logging

import uproot

logger = logging.getLogger(__name__)


def make_new_genie_injection(
    paths, simulation_config, *, detector_offset, detector=None, **kwargs
) -> None:
    """Validate a GENIE ROOT file before the simulation loop runs.

    Unlike ``LeptonInjector``, GENIE events are read directly from the ROOT file
    at construction time, so no output file is generated here. This function
    confirms the file is readable and logs a summary.

    Parameters
    ----------
    paths : GENIEInjectorPathsConfig
        Must have ``injection_file`` pointing to a gRooTracker ROOT file.
    simulation_config : GENIESimConfig
        Placement and seed settings (validated here for early error reporting).
    detector_offset : np.ndarray
        Centre of the detector in metres (unused here, accepted for interface
        compatibility).
    detector : Detector, optional
        Detector object. Required if ``simulation_config.placement='random'``.
    """
    genie_file = paths.injection_file
    placement = getattr(simulation_config, "placement", "fixed")

    if placement == "random" and detector is None:
        raise ValueError(
            "detector must be available when using GENIE placement='random'. "
            "Ensure Prometheus has a detector configured."
        )

    logger.info("Validating GENIE ROOT file: %s", genie_file)
    with uproot.open(genie_file) as f:
        if "gRooTracker" not in f:
            raise ValueError(f"{genie_file!r} does not contain the expected 'gRooTracker' tree")
        n_events = f["gRooTracker"].num_entries
    logger.info("GENIE ROOT file OK: %d events, placement=%s", n_events, placement)
