"""Decay-chain start times must accumulate across generations.

PROPOSAL restarts its clock at 0 for every propagation, so a decay product's
PROPOSAL time only covers its direct parent's flight. ``particle_from_proposal``
must add the parent's start time so that grandchildren (e.g. tau -> mu ->
decay) are timed from the event start, not from the intermediate particle's
start. These tests need no PROPOSAL/PPC installation.
"""

import numpy as np
import pytest

from prometheus.particle import PropagatableParticle, particle_from_proposal
from prometheus.utils.units import s_to_ns

_DIR_Z = np.array([0.0, 0.0, 1.0])


class _V:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _PPParticle:
    """Duck-typed PROPOSAL particle (energy in MeV, position in cm, time in s)."""

    def __init__(self, time_s, z_cm=49000.0):
        self.type = 13
        self.energy = 500_000.0
        self.position = _V(0.0, 0.0, z_cm)
        self.direction = _V(0.0, 0.0, 1.0)
        self.time = time_s


def _parent(time):
    return PropagatableParticle(15, 1e3, np.zeros(3), _DIR_Z, 0, None, time=time)


class TestFirstGeneration:
    def test_child_of_event_vertex_parent(self):
        # parent.time == 0, so the PROPOSAL clock coincides with the event
        # clock and the child's time is just the converted PROPOSAL time.
        pp = _PPParticle(time_s=1.634e-6)
        child = particle_from_proposal(pp, np.zeros(3), parent=_parent(time=0.0))
        assert child.time == pytest.approx(pp.time * s_to_ns)

    def test_zero_time_stays_zero(self):
        pp = _PPParticle(time_s=0.0)
        child = particle_from_proposal(pp, np.zeros(3), parent=_parent(time=0.0))
        assert child.time == 0.0


class TestGrandchild:
    def test_parent_start_time_is_added(self):
        parent_start = 1000.0  # ns
        pp = _PPParticle(time_s=1.634e-6)
        child = particle_from_proposal(pp, np.zeros(3), parent=_parent(time=parent_start))
        assert child.time == pytest.approx(parent_start + pp.time * s_to_ns)


class TestChainedGenerations:
    def test_times_accumulate_along_the_chain(self):
        # tau -> mu: the mu starts at the tau's decay vertex.
        pp_mu = _PPParticle(time_s=1.0e-6)
        mu = particle_from_proposal(pp_mu, np.zeros(3), parent=_parent(time=0.0))
        # mu -> decay product: each PROPOSAL propagation restarts at 0, so the
        # grandchild's total time must be the sum of both flight times.
        pp_decay = _PPParticle(time_s=2.5e-7, z_cm=56500.0)
        grandchild = particle_from_proposal(pp_decay, np.zeros(3), parent=mu)
        assert grandchild.time == pytest.approx((1.0e-6 + 2.5e-7) * s_to_ns)
        assert grandchild.parent is mu
