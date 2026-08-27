"""Olympus hits must carry the real module keys, not synthesized ones.

The old conversion assumed every string had the same number of OMs, string
IDs ran from 0 and OM IDs ran from 0 in file order. Any geo file that broke
those assumptions produced hits whose keys were not in the detector, and
serialization failed with ``ValueError: (s, o) is not in list``.
"""

import awkward as ak
import numpy as np
import pytest

from prometheus.detector import Detector, Medium
from prometheus.detector.module import Module
from prometheus.photon_propagation.olympus_photon_propagator import hits_from_olympus_result


def _detector(keys):
    # Place each string on its own vertical line so the geometry is valid.
    modules = [Module(np.array([100.0 * s, 0.0, 10.0 * o]), (s, o)) for s, o in keys]
    return Detector(modules, Medium.WATER)


def _one_hit_per_module(n_modules):
    return ak.Array([[10.0 * i] for i in range(n_modules)])


@pytest.mark.parametrize(
    "keys",
    [
        # 1-based string and OM numbering
        [(s, o) for s in range(1, 4) for o in range(1, 4)],
        # uneven strings
        [(0, o) for o in range(5)] + [(1, o) for o in range(2)],
        # OM-major file order
        [(s, o) for o in range(3) for s in range(2)],
        # non-contiguous string IDs
        [(s, o) for s in (3, 7, 42) for o in range(2)],
    ],
)
def test_hits_use_real_module_keys(keys):
    det = _detector(keys)
    hits = hits_from_olympus_result(det, _one_hit_per_module(len(keys)))

    assert [(h.string_id, h.om_id) for h in hits] == keys
    # Every hit must resolve on the detector, as serialization requires.
    for hit, mod in zip(hits, det.modules):
        assert det[(hit.string_id, hit.om_id)] is mod
    assert [h.time for h in hits] == [10.0 * i for i in range(len(keys))]


def test_multiple_hits_and_empty_modules():
    det = _detector([(0, 0), (0, 1), (1, 0)])
    hits = hits_from_olympus_result(det, ak.Array([[1.0, 2.0], [], [3.0]]))
    assert [(h.string_id, h.om_id, h.time) for h in hits] == [
        (0, 0, 1.0),
        (0, 0, 2.0),
        (1, 0, 3.0),
    ]


def test_none_result_gives_no_hits():
    det = _detector([(0, 0)])
    assert hits_from_olympus_result(det, None) == []


def test_module_count_mismatch_raises():
    det = _detector([(0, 0), (0, 1)])
    with pytest.raises(ValueError, match="module entries"):
        hits_from_olympus_result(det, ak.Array([[1.0]]))
