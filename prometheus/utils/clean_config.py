
def clean_config(config: dict) -> None:
    """Remove extraneous fields from config.

    Config dictionary is modified in place.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """
    # TODO move the actual config up a level
    for x in ["injection", "lepton propagator", "photon propagator"]:
        keys = [key for key in config[x].keys() if key not in ["name", config[x]["name"], "photon field name"]]
        for key in keys:
            del config[x][key]
