from typing import List, Optional

from ..hit import Hit


def parse_ppc(ppc_file: str) -> List[Hit]:
    """Parse a PPC output file into a list of hits.

    Handles both the legacy format::

        HIT string om time wavelength pth pph dth dph

    and the nextgen format produced when ``om.conf`` is present::

        HIT string dom_pmt time wavelength pth pph dth dph

    The format is auto-detected from the first HIT line.  A file may not mix
    formats; if a format inconsistency is detected a ``ValueError`` is raised.

    Parameters
    ----------
    ppc_file : str
        Path to the PPC output file.

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

            # In the HIT line `... pth pph dth dph`, tokens 5,6 are the photon
            # direction and tokens 7,8 are the impact position on the OM (in the
            # ppc engine, pth/pph build the direction `n` passed to getPMT, and
            # dth/dph build the position `r`). The Hit fields are named for the
            # physical quantity, so the direction pair fills photon_* and the
            # position pair fills om_* — not the token order.
            hits.append(
                Hit(
                    string_id=int(tokens[1]),
                    om_id=om_id,
                    time=float(tokens[3]),
                    wavelength=float(tokens[4]),
                    om_zenith=float(tokens[7]),
                    om_azimuth=float(tokens[8]),
                    photon_zenith=float(tokens[5]),
                    photon_azimuth=float(tokens[6]),
                    pmt_id=pmt_id,
                )
            )

    return hits
