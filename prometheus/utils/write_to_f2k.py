import numpy as np

from .units import SpeedOfLight, s_to_ns

PPC_MAGIC_Z = 1948.07


def f2k_angles(direction, official_ppc=False):
    """Return the f2k angles of a direction of travel.

    The official PPC reads degrees and the direction the particle came from, as
    in the f2000 format. The PPC bundled with Prometheus was changed to read
    radians and the direction of travel.

    Parameters
    ----------
    direction : np.ndarray
        Unit vector along the direction of travel.
    official_ppc : bool, optional
        Whether the file is for the official PPC.

    Returns
    -------
    tuple of float
        ``(theta, phi)`` as the chosen PPC reads them.
    """
    theta = np.arccos(direction[2])
    phi = np.arctan2(direction[1], direction[0])
    if not official_ppc:
        return theta, phi
    return 180 - np.degrees(theta), (np.degrees(phi) + 180) % 360


def serialize_particle(particle, output_f2k, official_ppc=False):
    """Write an MC header line for a particle to an open f2k file.

    Parameters
    ----------
    particle : Particle
        Particle whose energy and position are written.
    output_f2k : file-like object
        Open file to write to.
    official_ppc : bool, optional
        Whether the file is for the official PPC. See ``f2k_angles``.
    """
    offpos = particle.position
    theta, phi = f2k_angles(particle.direction, official_ppc)
    output_f2k.write(
        f"MC E {particle.e} x {offpos[0]} y {offpos[1]}"
        f" z {offpos[2] + PPC_MAGIC_Z} theta {theta} phi {phi}\n"
    )


def serialize_loss(loss, parent, output_f2k, official_ppc=False):
    """Write a ``TR`` track line for an energy loss to an open f2k file.

    Parameters
    ----------
    loss : Loss
        Energy loss to serialize.
    parent : Particle
        Parent particle that produced the loss; its direction and start
        time are used.
    output_f2k : file-like object
        Open file to write to.
    official_ppc : bool, optional
        Whether the file is for the official PPC. See ``f2k_angles``.
    """
    d = np.linalg.norm(loss.position - parent.position)
    offpos = loss.position
    theta, phi = f2k_angles(parent.direction, official_ppc)
    c = SpeedOfLight
    c /= s_to_ns
    # Offset by the parent's start time so decay-product light is timed from the
    # decay vertex (reached at ~L_decay/c), not from t=0. The primary has
    # time=0, leaving its own losses unchanged.
    dt = parent.time + d / c
    line = (
        f"TR 0 {0} {loss} {offpos[0]} {offpos[1]} {offpos[2] + PPC_MAGIC_Z}"
        f" {theta} {phi} 0 {loss.e} {dt} \n"
    )
    output_f2k.write(line)


def serialize_to_f2k(particle, fname, official_ppc=False):
    """Create an f2k file from given events.

    The f2k file is modified in place, the output can be passed to ppc to propagate the photons.

    Parameters
    ----------
    particle : Particle
        The particle to serialize.
    fname : str
        File name to write the data to.
    official_ppc : bool, optional
        Whether the file is for the official PPC. See ``f2k_angles``.

    Notes
    -----
    Details of the output format can be found here:
    <https://www.zeuthen.desy.de/~steffenp/f2000/>.

    In a nutshell this format is as follows:

    ``TR int int name x y z theta phi length energy time``

    - ``TR`` stands for track and is a character constant.
    - The first two ``int`` values are not used by ppc and are just for book keeping.
    - The ``name`` column specifies the track type.
        - Possible values are: "amu+," "amu-," and "amu" for muons,
        "delta," "brems," "epair," "e+,", "e-," and "e" for electromagnetic cascades,
        and "munu" and "hadr" for hadronic cascades.
    - ``x``, ``y`` and ``z`` are the vector components of the track's initial position in meters.
    - The quantities ``theta`` and ``phi`` are the track's theta and phi angle.
      Their units and sense depend on the PPC version, see ``f2k_angles``.
    - ``length`` is the length of the track in meters.
        - It is only required for muons because cascades are treated as point-like sources.
    - ``energy`` is the track's initial energy in GeV.
    - ``time`` is the track's initial time in nanoseconds.
    """
    index = 0
    with open(fname, "w") as output_f2k:
        output_f2k.write(f"EM {index} 1 0 0 0 0 \n")
        serialize_particle(particle, output_f2k, official_ppc)
        for loss in particle.losses:
            serialize_loss(loss, particle, output_f2k, official_ppc)
        for child in particle.children:
            serialize_particle(child, output_f2k, official_ppc)
            for loss in child.losses:
                serialize_loss(loss, child, output_f2k, official_ppc)
        output_f2k.write("EE\n")
