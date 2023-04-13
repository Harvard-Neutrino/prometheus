Module prometheus.utils.geo_utils
=================================

Functions
---------

    
`from_geo(fname)`
:   Returns positions, keys, and medium from detector geometry file

    
`geo_from_coords(coords, out_path, tol=0.5, medium='ice', dom_radius=30)`
:   Generates a detector geometry file
    Parameters
    __________
    
        coords:
            nx3 array of modules coordinates
        out_path:
            File path to write to
        tol:
            Tolerance for grouping DOMs in a string

    
`geo_from_f2k(fname, out_path, medium='ice', dom_radius=30)`
:   Generates a detector geo file from an f2k

    
`get_cylinder(coords, epsilon=5)`
:   

    
`get_endcap(coords, is_ice)`
:   

    
`get_injection_radius(coords, is_ice)`
:   

    
`get_volume(coords, is_ice)`
:   

    
`get_xyz(fname)`
:   

    
`offset(coords)`
: