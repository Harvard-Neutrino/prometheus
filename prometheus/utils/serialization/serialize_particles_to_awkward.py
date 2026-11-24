import awkward as ak
import numpy as np

from prometheus.detector import Detector
from prometheus.injection.injection import Injection
from prometheus.photon_propagation.hit import Hit

from .accumulate_hits import accumulate_hits

#: Output modes understood by this module, in order of increasing verbosity.
_VALID_MODES = ("minimal", "standard", "extended")


def _hit_cartesian(hit: Hit, module):
    """Reconstruct the Cartesian impact point in the module-centered frame.

    Mirrors the reconstruction performed in PPC's ``f2k.cxx``.  Coordinates
    are in metres, relative to the module centre.

    Parameters
    ----------
    hit : Hit
        Hit whose ``om_zenith`` / ``om_azimuth`` encode the impact point.
    module : Module
        Module that was struck; provides ``Rr`` and ``Rz``.

    Returns
    -------
    tuple of float
        ``(hit_x, hit_y, hit_z)`` in metres.
    """
    F = module.Rz / module.Rr
    dth = hit.om_zenith
    dph = hit.om_azimuth

    if module.Rz < 0:  # cylindrical
        rx = np.cos(dph)
        ry = np.sin(dph)
    else:  # spherical or spheroidal
        rx = np.sin(dth) * np.cos(dph)
        ry = np.sin(dth) * np.sin(dph)
    rz = F * np.cos(dth)

    return -module.Rr * rx, -module.Rr * ry, -module.Rr * rz


def serialize_particles_to_awkward(
    det: Detector,
    injection: Injection,
    output_mode: str = "minimal",
):
    """Serialize hit information from all injection events into an ``awkward.Array``.

    Parameters
    ----------
    det : Detector
        Prometheus detector used to look up module positions and shapes.
    injection : Injection
        Injection object containing the simulated events.
    output_mode : str, optional
        Controls which hit fields are written to the output array:

        ``"minimal"``
            ``sensor_pos_{x,y,z}``, ``string_id``, ``sensor_id``, ``t``,
            ``id_idx``.  This is the default and matches the previous behaviour.
        ``"standard"``
            Everything in ``"minimal"`` plus ``wavelength``,
            ``photon_zenith``, ``photon_azimuth``, ``om_zenith``,
            ``om_azimuth``, and ``pmt_id``.
        ``"extended"``
            Everything in ``"standard"`` plus ``hit_x``, ``hit_y``, ``hit_z``
            (Cartesian impact point in the module-centred frame, metres).

    Returns
    -------
    ak.Array or None
        Array with the requested fields for each event, or ``None`` if no
        hits were recorded.

    Raises
    ------
    ValueError
        If ``output_mode`` is not one of the recognised values.
    """
    if output_mode not in _VALID_MODES:
        raise ValueError(f"output_mode must be one of {_VALID_MODES!r}, got {output_mode!r}")

    all_hits = []
    for injection_event in injection:
        all_hits.append(accumulate_hits(injection_event.final_states))

    if not any(len(h) > 0 for h in all_hits):
        return None

    xyz = [
        np.array([det[(h.string_id, h.om_id)].pos for h, _ in event_hits]).transpose()
        for event_hits in all_hits
    ]

    outdict = {}
    for idx, var in enumerate("x y z".split()):
        outdict[f"sensor_pos_{var}"] = [x[idx] if x.shape[0] > 0 else np.array([]) for x in xyz]

    # Always-present fields
    hit_functions = [
        ("string_id", lambda x: [[h.string_id for h, _ in ev] for ev in x]),
        ("sensor_id", lambda x: [[h.om_id for h, _ in ev] for ev in x]),
        ("t", lambda x: [[h.time for h, _ in ev] for ev in x]),
        ("id_idx", lambda x: [[idx for _, idx in ev] for ev in x]),
    ]

    if output_mode in ("standard", "extended"):
        hit_functions += [
            ("wavelength", lambda x: [[h.wavelength for h, _ in ev] for ev in x]),
            ("photon_zenith", lambda x: [[h.photon_zenith for h, _ in ev] for ev in x]),
            ("photon_azimuth", lambda x: [[h.photon_azimuth for h, _ in ev] for ev in x]),
            ("om_zenith", lambda x: [[h.om_zenith for h, _ in ev] for ev in x]),
            ("om_azimuth", lambda x: [[h.om_azimuth for h, _ in ev] for ev in x]),
            ("pmt_id", lambda x: [[h.pmt_id for h, _ in ev] for ev in x]),
        ]

    for field, fxn in hit_functions:
        outdict[field] = fxn(all_hits)

    if output_mode == "extended":
        hit_x_list, hit_y_list, hit_z_list = [], [], []
        for event_hits in all_hits:
            xs, ys, zs = [], [], []
            for h, _ in event_hits:
                if h.om_zenith is not None and h.om_azimuth is not None:
                    module = det[(h.string_id, h.om_id)]
                    hx, hy, hz = _hit_cartesian(h, module)
                else:
                    hx = hy = hz = None
                xs.append(hx)
                ys.append(hy)
                zs.append(hz)
            hit_x_list.append(xs)
            hit_y_list.append(ys)
            hit_z_list.append(zs)
        outdict["hit_x"] = hit_x_list
        outdict["hit_y"] = hit_y_list
        outdict["hit_z"] = hit_z_list

    return ak.Array(outdict)
