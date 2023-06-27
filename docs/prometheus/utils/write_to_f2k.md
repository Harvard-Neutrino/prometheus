Module prometheus.utils.write_to_f2k
====================================

Functions
---------

    
`serialize_loss(loss, parent, output_f2k)`
:   

    
`serialize_particle(particle, output_f2k)`
:   

    
`serialize_to_f2k(particle, fname)`
:   output_f2k: file name where the data will be written to
    This format output can be passed to PPC to propagate the photons.
    Details of the format can be found here:
    https://www.zeuthen.desy.de/~steffenp/f2000/
    In a nutshell this format is as follows:
    
    'TR int int name x y z theta phi length energy time.'
    
    - TR stands for track and is a character constant.
    - The first two integer values are not used by ppc and are just for book keeping.
    - The name column specifies the track type.
        Possible values are: "amu+," "amu-," and "amu" for muons,
        "delta," "brems," "epair," "e+,", "e-," and "e" for electromagnetic cascades,
        and "munu" and "hadr" for hadronic cascades.
    - x, y and z are the vector components of the track's initial position in meters.
    - The quantities theta and phi are the track's theta and phi angle in degrees, respectively.
        length is the length of the track in meter.
        It is only required for muons because cascades are treated as point-like sources.
        energy is the track's initial energy in GeV.
    - Time is the track's initial time in nanoseconds.