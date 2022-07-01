import numpy as np
from .convert_loss_name import convert_loss_name
from .units import SpeedOfLight, s_to_ns

def serialize_particle(particle, output_f2k, offset):
    offpos = particle.position + offset
    offpos[2] += 1948.07
    theta = np.arccos(particle.direction[2])
    phi = np.arctan2(particle.direction[1], particle.direction[0])
    output_f2k.write(
        f'MC E {particle.e} x {offpos[0]} y {offpos[1]} z {offpos[2]} theta {theta} phi {phi}\n'
    )

def serialize_loss(loss, parent, output_f2k, offset):
    d = np.linalg.norm(loss.position - parent.position)
    offpos = loss.position + offset
    offpos[2] += 1948.07
    theta = np.arccos(parent.direction[2])
    phi = np.arctan2(parent.direction[1], parent.direction[0])
    c = SpeedOfLight
    c /= s_to_ns
    dt = d / c
    line = f'TR 0 {0} {str(loss)} {offpos[0]} {offpos[1]} {offpos[2]} {theta} {phi} 0 {loss.e} {dt} \n'
    #line = f'TR 0 {loss.int_type} {str(loss)} {offpos[0]} {offpos[1]} {offpos[2]} {theta} {phi} 0 {loss.e} {dt} \n'
    output_f2k.write(line)



# Create an f2k file from given events
def serialize_to_f2k(particle, fname, det_offset):
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
        output_f2k.write(f'EM {index} 1 {det_offset[2]} 0 0 0 \n')
        serialize_particle(particle, output_f2k, det_offset)
        for loss in particle.losses:
            serialize_loss(loss, particle, output_f2k, det_offset)
        for child in particle.children:
            serialize_particle(child, output_f2k, det_offset)
            for loss in child.losses:
                serialize_loss(loss, child, output_f2k, det_offset)
        output_f2k.write('EE\n')
