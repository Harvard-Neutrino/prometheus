Module prometheus.detector.detector_factory
===========================================

Functions
---------

    
`detector_from_geo(geofile: str, efficiency: float = 0.2, noise_rate: float = 1) ‑> prometheus.detector.detector.Detector`
:   Make a detector from a Prometheus geo file
    
    params
    ______
    geofile: Geofile to read from
    efficiency: Quantum efficiency of OMs
    noise_rate: Noise rate of OMs in Hz
    
    returns
    _______
    detector: Prometheus detector object

    
`make_grid(n_side: int, dist: float, n_z: int, dist_z: float, z_cent: float, rng: Union[int, None, numpy.random.mtrand.RandomState] = 1337, baseline_noise_rate: float = 1000.0, efficiency: float = 0.2) ‑> prometheus.detector.detector.Detector`
:   Build a square detector grid. Strings of detector modules are placed 
    on a square grid, with the number of strings per side, number of modules 
    per string, and z-spacing on a string set by input. The noise rate for 
    each module is randomly sampled from a gamma distribution. The random
    state may be set by input
    
    params
    ______
    n_side: number of strings per side
    dist: spacing between strings along principal axes
    n_z: number of modules on the line
    dist_z: vertical spacing between modules
    z_cent: z-position of the center of the detector
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs
    
    returns
    _______
    det: Orthogonal Prometheus Detector

    
`make_hex_grid(n_side: int, dist: float, n_z: int, dist_z: float, z_cent: float, baseline_noise_rate: float = 1000.0, rng: Union[int, None, numpy.random.mtrand.RandomState] = 1337, efficiency: float = 0.2) ‑> prometheus.detector.detector.Detector`
:   Build a hex detector grid. Strings of detector modules are placed on a hexagonal
    grid with number of OMs per string and distance between these modules set by input.
    The vertical center of the detector is at z_cent.
    The noise rate for each module is randomöy sampled from a gamma distribution.
    
    params
    ______
    n_side: number of strings per side of hexagon
    dist: insterstring spacing in meters
    n_z: number of modules on the line
    dist_z: vertical spacing between modules
    z_cent: z-position of the center of the line
    line_id: integer identifier of the line
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs
    
    returns
    _______
    det: Hexagonal Prometheus detector

    
`make_line(x: float, y: float, n_z: int, dist_z: float, z_cent: float, line_id: int, rng: numpy.random.mtrand.RandomState = 1337, baseline_noise_rate: float = 1000.0, efficiency: float = 0.2) ‑> prometheus.detector.detector.Detector`
:   Make a line of detector modules. The modules share the same (x, y) coordinate and 
    are spaced along the z-direction. This detector will be symetrically spaced about z=z_cent
    
    params
    ______
    x: x-position of the line
    y: y-position of the line
    n_z: number of modules on the line
    dist_z: vertical spacing between modules
    z_cent: z-position of the center of the line
    line_id: integer identifier of the line
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs
    
    returns
    _______
    modules: list of modules. This may be then be used to construct a Prometheus
        Detector if a single line detector is desired

    
`make_rhombus(side_len, n_z, dist_z, z_cent, rng: Union[int, None, numpy.random.mtrand.RandomState] = 1337, baseline_noise_rate: float = 1000.0, efficiency: float = 0.2) ‑> prometheus.detector.detector.Detector`
:   Make a rhombus detector
    
    params
    ______
    side_len: length of the rhombus in meters
    n_z: number of modules on the line
    dist_z: vertical spacing between modules in meters
    z_cent: z-position of the center of the detector in meters
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs
    
    returns
    _______
    det: A rhombus detector

    
`make_triang(side_len, n_z, dist_z, z_cent, rng: numpy.random.mtrand.RandomState = 1337, baseline_noise_rate: float = 1000.0, efficiency: float = 0.2) ‑> prometheus.detector.detector.Detector`
:   Build a triangular detector grid. Strings of detector modules are placed 
    on a the corners of a equilateral triangle, with input side length,
    number of modules per string, and z-spacing on a string set by input.
    The noise rate for each module is randomly sampled from a gamma distribution. 
    The random state may be set by input
    
    params
    ______
    side_len: length of the triangle in meters
    n_z: number of modules on the line
    dist_z: vertical spacing between modules
    z_cent: z-position of the center of the detector
    [rng]: How to set numpy random state. If a np.random.RandomState instance is passed, 
        that will be used. If int or None, random state will be 
        np.random.RandomState(rng). Anything else will raise error
    [baseline_noise_rate]: Baseline dark noise rate for the OMs in GHz
    [efficiency]: quantum efficiency of the OMs
    
    returns
    _______
    det: a triangular detector

    
`parse_rng(rng: Union[int, None, numpy.random.mtrand.RandomState]) ‑> numpy.random.mtrand.RandomState`
:   Helps determine random number generation state from input
    
    params
    ______
    rng: rng generator to make sense of
    
    returns
    _______
    rng: np.random.RandomState
    
    raises
    ______
    InvalidRNGError: If we don't know how to handle the input rng

    
`read_medium(geofile) ‑> Optional[prometheus.detector.medium.Medium]`
:   Figures out detector medium from geofile
    
    params
    ______
    geofile: Detector geometry file
    
    returns
    _______
    medium: Medium for the detector

Classes
-------

`InvalidRNGError(rng)`
:   Raised when rng specification can't be parsed

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException