Module prometheus.injection.genie_parser
========================================

Functions
---------

    
`angle(v1: <built-in function array>, v2: <built-in function array>) ‑> float`
:   Calculates the angle between two vectors in radians
    
    Parameters
    ----------
    v1: np.array
        vector 1
    v2: np.array
        vector 2
    
    Returns
    -------
    angle: float
        The calculates angle in radians

    
`final_parser(parsed_events: pandas.core.frame.DataFrame) ‑> pandas.core.frame.DataFrame`
:   fetches the final states
    
    Parameters
    ----------
    parsed_events : pd.DataFrame
        The parsed events
    
    Returns
    -------
    pd.DataFrame
        The inital + final state info

    
`genie2prometheus(parsed_events: pandas.core.frame.DataFrame)`
:   reformats parsed GENIE events into a usable format for PROMETHEUS
    NOTES: Create a standardized scheme function. This could then be used as an interface to PROMETHEUS
    for any injector. E.g. a user would only need to create a function to translate their injector output to the scheme format.
    
    Parameters
    ----------
    parsed_events: pd.DataFrame
        Dataframe object containing all the relevant (and additional) information needed by PROMETHEUS
    
    Returns
    -------
    particles: pd.DataFrame
        Fromatted data set with values which can be used directly.
    injection: pd.Dataframe
        The injected particle in the same format

    
`genie_loader(filepath: str) ‑> pandas.core.frame.DataFrame`
:   Loads and parses GENIE data
    
    Parameters
    ----------
    filepath: str
        Path to the GENIE file
    
    Returns
    -------
    pd.DataFrame
        The parsed genie data

    
`genie_parser(events) ‑> pandas.core.frame.DataFrame`
:   function to fetch the relevant information from genie events (in rootracker format)
    
    Parameters
    ----------
    events : dict
        The genie events
    
    Returns
    -------
    pd.DataFrame
        Data frame containing the relevant information

    
`p2azimuthAndzenith(p: <built-in function array>)`
:   converts a momentum vector to azimuth and zenith angles
    
    Parameters
    ----------
    p: np.array
        The 3d momentum
    
    Returns
    -------
    float, float:
        The azimuth and zenith angles in radians.