from typing import List, Optional

from ..hit import Hit


def parse_ppc(ppc_file: str, official_ppc: bool = False) -> List[Hit]:
    """Parse a PPC output file into a list of hits.

    Handles both the legacy format, here from the bundled PPC::

        HIT string om time wavelength dth dph pth pph

    and the nextgen format produced when ``om.conf`` is present::

        HIT string dom_pmt time wavelength pth pph dth dph

    Here ``d`` is the impact point on the module and ``p`` the photon direction.
    The bundled PPC writes the former first. The official PPC, which is the only
    one with nextgen output, writes the latter first in both formats.

    The format is auto-detected from the first HIT line.  A file may not mix
    formats; if a format inconsistency is detected a ``ValueError`` is raised.

    Parameters
    ----------
    ppc_file : str
        Path to the PPC output file.
    official_ppc : bool, optional
        Whether the file was written by the official PPC. Only matters for the
        legacy format.

    Returns
    -------
    list of Hit
        Parsed hit objects.  ``Hit.pmt_id`` is ``None`` in legacy mode and an
        integer PMT index in nextgen mode.

    Raises
    ------
    ValueError
        If the file mixes legacy and nextgen HIT lines.
    """
    hits: List[Hit] = []
    nextgen: Optional[bool] = None

    with open(ppc_file) as ppc_out:
        for line in ppc_out:
            if "HIT" not in line:
                continue
            tokens = line.split()
            line_is_nextgen = "_" in tokens[2]

            if nextgen is None:
                nextgen = line_is_nextgen
            elif nextgen != line_is_nextgen:
                raise ValueError("PPC output mixes legacy and nextgen HIT formats in the same file")

            if nextgen:
                dom_str, pmt_str = tokens[2].split("_", 1)
                om_id = int(dom_str)
                pmt_id: Optional[int] = int(pmt_str)
            else:
                om_id = int(tokens[2])
                pmt_id = None

            if nextgen or official_ppc:
                photon_angles = tokens[5:7]
                om_angles = tokens[7:9]
            else:
                om_angles = tokens[5:7]
                photon_angles = tokens[7:9]

            hits.append(
                Hit(
                    string_id=int(tokens[1]),
                    om_id=om_id,
                    time=float(tokens[3]),
                    wavelength=float(tokens[4]),
                    om_zenith=float(om_angles[0]),
                    om_azimuth=float(om_angles[1]),
                    photon_zenith=float(photon_angles[0]),
                    photon_azimuth=float(photon_angles[1]),
                    pmt_id=pmt_id,
                )
            )

    return hits
