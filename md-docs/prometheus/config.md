Module prometheus.config
========================

Classes
-------

`ConfigClass(*args, **kwargs)`
:   The configuration class. This is used
    by the package for all parameter settings. If something goes wrong
    its usually here.
    Parameters
    ----------
    config : dic
        The config dictionary
    Returns
    -------
    None

    ### Ancestors (in MRO)

    * builtins.dict

    ### Methods

    `from_dict(self, user_dict: Dict[Any, Any]) ‑> None`
    :   Creates a config from dictionary
        Parameters
        ----------
        user_dict : dic
            The user dictionary
        Returns
        -------
        None

    `from_yaml(self, yaml_file: str) ‑> None`
    :   Update config with yaml file
        Parameters
        ----------
        yaml_file : str
            path to yaml file
        Returns
        -------
        None