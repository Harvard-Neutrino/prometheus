import numpy as np
from .convert_loss_name import convert_loss_name
from .units import SpeedOfLight, s_to_ns

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

## !! uncomment following block if you wish to see logger line (TR f2k print line) in this file: !!

# if not logger.handlers:
#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(logging.DEBUG)  # Make sure handler allows DEBUG
#     formatter = logging.Formatter('%(levelname)s: %(message)s')
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)

PPC_MAGIC_Z = 1948.07

def serialize_particle(particle, output_f2k):
    offpos = particle.position
    theta = np.arccos(particle.direction[2])
    phi = np.arctan2(particle.direction[1], particle.direction[0])
    output_f2k.write(
        f'MC E {particle.e} x {offpos[0]} y {offpos[1]} z {offpos[2] + PPC_MAGIC_Z} theta {theta} phi {phi}\n'
    )

def serialize_loss(loss, parent, output_f2k):
    d = np.linalg.norm(loss.position - parent.position)
    offpos = loss.position
    theta = np.arccos(parent.direction[2])
    phi = np.arctan2(parent.direction[1], parent.direction[0])
    track_length = loss.track_length ## non zero track for amu- loss type
    c = SpeedOfLight
    c /= s_to_ns
    dt = d / c
    line = f'TR 0 {0} {loss} {offpos[0]} {offpos[1]} {offpos[2] + PPC_MAGIC_Z} {theta} {phi} {track_length} {loss.e} {dt} \n'
    logger.debug(f'TR 0 {0} {loss} {offpos[0]} {offpos[1]} {offpos[2] + PPC_MAGIC_Z} {theta} {phi} {track_length} {loss.e} {dt}') ## just for logging purposes
    output_f2k.write(line)

# Create an f2k file from given events
def serialize_to_f2k(particle, fname):
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
    index = 0
    with open(fname, "w") as output_f2k:
        output_f2k.write(f'EM {index} 1 0 0 0 0 \n')
        serialize_particle(particle, output_f2k)
        for loss in particle.losses:
            serialize_loss(loss, particle, output_f2k)
        for child in particle.children:
            serialize_particle(child, output_f2k)
            for loss in child.losses:
                serialize_loss(loss, child, output_f2k)
        output_f2k.write('EE\n')
