#!/usr/bin/env python3
"""A/B-compare two PPC HIT outputs to check the engine swap didn't change output.

Feed it two PPC stdout capture files produced by running the *baseline* binary
and the *candidate* binary on the SAME f2k input and SAME tables:

    baseline/ppc_cpu <device> < events.f2k > baseline_hits.txt
    candidate/ppc_cpu <device> < events.f2k > candidate_hits.txt
    python scripts/ab_compare_hits.py baseline_hits.txt candidate_hits.txt

Compares pooled hit-level distributions (count, per-OM occupancy, arrival time,
wavelength). PPC seeds its RNG from wall-clock time unless built with DTMN, so
use many events and read this as a statistical (not bit-exact) comparison.

HIT line formats (tokens after "HIT"):
  legacy/nextgen-single:  string om        time wv pth pph dth dph
  nextgen multi-PMT:      string om_pmt    time wv pth pph dth dph
"""

import sys
from collections import Counter

try:
    from scipy.stats import ks_2samp

    def _ks(a, b):
        return ks_2samp(a, b).pvalue
except Exception:  # scipy optional; fall back to a coarse mean/std check
    ks_2samp = None

    def _ks(a, b):
        return None


def parse_hits(path):
    """Return dict of parallel lists: om_key, time, wavelength, pmt_id."""
    om_key, time, wv, pmt = [], [], [], []
    with open(path) as f:
        for line in f:
            if "HIT" not in line:
                continue
            t = line.split()
            om_tok = t[2]
            if "_" in om_tok:
                dom, p = om_tok.split("_", 1)
                pmt.append(int(p))
            else:
                dom = om_tok
                pmt.append(None)
            om_key.append((int(t[1]), int(dom)))
            time.append(float(t[3]))
            wv.append(float(t[4]))
    return {"om_key": om_key, "time": time, "wavelength": wv, "pmt_id": pmt}


def _mean(x):
    return sum(x) / len(x) if x else float("nan")


def occupancy_corr(a_keys, b_keys):
    """Pearson correlation of per-OM hit counts between the two runs."""
    ca, cb = Counter(a_keys), Counter(b_keys)
    keys = set(ca) | set(cb)
    xa = [ca.get(k, 0) for k in keys]
    xb = [cb.get(k, 0) for k in keys]
    n = len(keys)
    if n < 2:
        return float("nan")
    ma, mb = _mean(xa), _mean(xb)
    cov = sum((xa[i] - ma) * (xb[i] - mb) for i in range(n))
    va = sum((v - ma) ** 2 for v in xa) ** 0.5
    vb = sum((v - mb) ** 2 for v in xb) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ab_compare_hits.py baseline_hits.txt candidate_hits.txt", file=sys.stderr)
        return 2

    a = parse_hits(sys.argv[1])
    b = parse_hits(sys.argv[2])

    na, nb = len(a["time"]), len(b["time"])
    ratio = nb / na if na else float("nan")
    print(f"total hits:      baseline={na}  candidate={nb}  ratio={ratio:.4f}")
    print(f"mean time [ns]:  baseline={_mean(a['time']):.2f}  candidate={_mean(b['time']):.2f}")
    mwa, mwb = _mean(a["wavelength"]), _mean(b["wavelength"])
    print(f"mean wv   [nm]:  baseline={mwa:.2f}  candidate={mwb:.2f}")
    print(f"per-OM occupancy correlation: {occupancy_corr(a['om_key'], b['om_key']):.5f}")
    if ks_2samp is not None:
        print(f"KS p (time):       {_ks(a['time'], b['time']):.4f}")
        print(f"KS p (wavelength): {_ks(a['wavelength'], b['wavelength']):.4f}")
    else:
        print("(install scipy for KS tests)")

    npmt_a = len({p for p in a["pmt_id"] if p is not None})
    npmt_b = len({p for p in b["pmt_id"] if p is not None})
    print(f"distinct pmt_id:  baseline={npmt_a}  candidate={npmt_b}  (nextgen/multi-PMT only)")

    print(
        "\nInterpret: for an ICE regression run, expect ratio ~1, occupancy corr "
        ">0.999 (cascades), and KS p not tiny. See docs/ppc_nextgen_validation.md "
        "for per-topology thresholds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
