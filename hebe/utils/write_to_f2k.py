import numpy as np
from .convert_loss_name import convert_loss_name

# Create an f2k file from given events
def write_to_f2k_format(mCPEnergyLosses, output_f2k, index):
    '''
        output_f2k: file name where the data will be written to
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
    '''
    # TODO put this in units or some shit
    # Useful constants
    km_to_m = 1e3
    MeV_to_GeV = 1e-3

    # Speed of light [m/s]
    SpeedOfLight = 299792458

    # Useful unit conversions
    meter_to_cm = 1e2
    km_to_cm = 1e5
    km_to_m = 1e3
    cm_to_m = 1e-2
    GeV_to_MeV = 1e3
    MeV_to_GeV = 1e-3
    s_to_ns = 1e9

    # TODO move this to the config file?
    ic_depth = 1970 # m
    costh, phi, E, iposition, losses = mCPEnergyLosses
    output_f2k.write('EM {i} 1 {depth_in_meters} 0 0 0 \n'.format(i=index, depth_in_meters=ic_depth * km_to_m))
    output_f2k.write('MC E {E} x {x} y {y} z {z} theta {theta} phi {phi}\n'.format(
        E=E,
        x=iposition[0],
        y=iposition[1],
        z=iposition[2],
        theta=np.arccos(costh),
        phi=phi)
    )
    for ii, type_of_loss in enumerate(losses.keys()):
        for loss in losses[type_of_loss]:
            dr = iposition - loss[1]
            d = np.sqrt(np.square(dr[0]) + np.square(dr[1]) + np.square(dr[2]))
            # Convert speed of light to m/ns
            c = SpeedOfLight
            c /= s_to_ns
            dt = d / c
            magic_format = 'TR {i} {ii} {type_of_loss} {x} {y} {z} {theta} {phi} 0 {ee} {t} \n'.format(
                i=index,
                ii=ii,
                type_of_loss=convert_loss_name(type_of_loss),
                x=loss[1][0],
                y=loss[1][1],
                z=loss[1][2] + ic_depth,
                theta=np.arccos(costh),
                phi=phi,
                ee=10**loss[0],
                t=dt
            )
            output_f2k.write(magic_format)
    output_f2k.write('EE\n')

