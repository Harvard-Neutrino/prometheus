from typing import List

from ..hit import Hit


def parse_ppc(ppc_file: str) -> List[Hit]:
    """Parse ppc output file into a list of hits.

    Parameters
    ----------
    ppc_file : str
        Path to the ppc output file.

    Returns
    -------
    hits : list of Hit
        List of Hit objects parsed from the ppc output.
    """
    hits = []
    with open(ppc_file) as ppc_out:
        for line in ppc_out:
            if "HIT" not in line:
                continue
            tokens = line.split()
            hit = Hit(
                int(tokens[1]),
                int(tokens[2]),
                float(tokens[3]),
                float(tokens[4]),
                float(tokens[5]),
                float(tokens[6]),
                float(tokens[7]),
                float(tokens[8]),
            )
            hits.append(hit)
    return hits
