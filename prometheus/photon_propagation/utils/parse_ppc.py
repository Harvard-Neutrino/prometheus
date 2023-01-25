from typing import List

from ..hit import Hit

def parse_ppc(ppc_file: str) -> List[Hit]:
    hits = []
    with open(ppc_file) as ppc_out:
        for line in ppc_out:
            if "HIT" not in line:
                continue
            l = line.split()
            hit = Hit(
                int(l[1]), int(l[2]), float(l[3]), float(l[4]),
                float(l[5]), float(l[6]), float(l[7]), float(l[8])
            )
            hits.append(hit)
    return hits
