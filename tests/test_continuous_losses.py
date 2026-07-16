"""Unit tests for continuous-energy-loss handling in new_proposal_losses.

PROPOSAL v7's ``Secondaries.stochastic_losses()`` never contains
ContinuousEnergyLoss (type 1000000008) entries; continuous losses are only
available via ``Secondaries.continuous_losses()``. These tests fake the
PROPOSAL objects to check that the continuous energy actually ends up in the
1000000008 point losses smeared along the track.
"""

import numpy as np
import pytest

from prometheus.lepton_propagation.new_proposal_lepton_propagator import (
    new_proposal_losses,
)

CONTINUOUS_TYPE = 1000000008
EPAIR_TYPE = 1000000004


class FakeVector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeStochasticLoss:
    def __init__(self, loss_type, energy_mev, position_cm):
        self.type = loss_type
        self.energy = energy_mev
        self.position = FakeVector(*position_cm)


class FakeContinuousLoss:
    def __init__(self, energy_mev):
        self.energy = energy_mev


class FakeSecondaries:
    def __init__(self, stochastic, continuous, distances_cm):
        self._stochastic = stochastic
        self._continuous = continuous
        self._distances_cm = distances_cm

    def stochastic_losses(self):
        return self._stochastic

    def continuous_losses(self):
        return self._continuous

    def decay_products(self):
        return []

    def track_propagated_distances(self):
        return self._distances_cm


class FakePropagator:
    def __init__(self, secondaries):
        self._secondaries = secondaries

    def propagate(self, init_state, distance_cm):
        return self._secondaries


class FakeParticle:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.direction = np.array([0.0, 0.0, 1.0])
        self.e = 1e3
        self.time = 0.0
        self.losses = []
        self.children = []


COORDINATE_SHIFT = np.array([0.0, 0.0, 100.0])
DETECTOR_CENTER = np.array([0.0, 0.0, 0.0])
R_INICE = 500.0


def run_losses(secondaries):
    particle = FakeParticle()
    new_proposal_losses(
        FakePropagator(secondaries),
        particle,
        padding=100.0,
        r_inice=R_INICE,
        detector_center=DETECTOR_CENTER,
        coordinate_shift=COORDINATE_SHIFT,
    )
    return particle


def prometheus_to_proposal_cm(position_m):
    return (np.asarray(position_m) + COORDINATE_SHIFT) * 1e2


def test_continuous_losses_smeared_along_track():
    secondaries = FakeSecondaries(
        stochastic=[],
        continuous=[FakeContinuousLoss(500.0), FakeContinuousLoss(1500.0)],
        distances_cm=[100.0 * 1e2],
    )
    particle = run_losses(secondaries)

    deltas = [loss for loss in particle.losses if loss.int_type == CONTINUOUS_TYPE]
    assert len(deltas) == 100
    assert sum(loss.e for loss in deltas) == pytest.approx(2.0)
    assert all(loss.e == pytest.approx(0.02) for loss in deltas)
    np.testing.assert_allclose(deltas[0].position, particle.position)
    np.testing.assert_allclose(deltas[1].position, particle.position + particle.direction)


def test_continuous_sum_independent_of_stochastic_losses():
    # PROPOSAL never puts type-1000000008 entries in stochastic_losses(), so
    # the continuous energy must come from continuous_losses() alone.
    secondaries = FakeSecondaries(
        stochastic=[
            FakeStochasticLoss(EPAIR_TYPE, 3000.0, prometheus_to_proposal_cm([0.0, 0.0, 10.0]))
        ],
        continuous=[FakeContinuousLoss(2000.0)],
        distances_cm=[100.0 * 1e2],
    )
    assert not any(loss.type == CONTINUOUS_TYPE for loss in secondaries.stochastic_losses())
    particle = run_losses(secondaries)

    continuous_total = sum(loss.e for loss in particle.losses if loss.int_type == CONTINUOUS_TYPE)
    assert continuous_total == pytest.approx(2.0)
    assert continuous_total > 0.0


def test_stochastic_losses_converted_and_filtered_by_r_inice():
    inside = FakeStochasticLoss(EPAIR_TYPE, 3000.0, prometheus_to_proposal_cm([0.0, 0.0, 10.0]))
    outside = FakeStochasticLoss(
        EPAIR_TYPE, 5000.0, prometheus_to_proposal_cm([0.0, 0.0, R_INICE + 1.0])
    )
    secondaries = FakeSecondaries(
        stochastic=[inside, outside],
        continuous=[],
        distances_cm=[100.0 * 1e2],
    )
    particle = run_losses(secondaries)

    stochastic = [loss for loss in particle.losses if loss.int_type == EPAIR_TYPE]
    assert len(stochastic) == 1
    assert stochastic[0].e == pytest.approx(3.0)
    np.testing.assert_allclose(stochastic[0].position, [0.0, 0.0, 10.0], atol=1e-9)


def test_zero_length_track_fallback():
    secondaries = FakeSecondaries(
        stochastic=[],
        continuous=[FakeContinuousLoss(2000.0)],
        distances_cm=[0.0],
    )
    particle = run_losses(secondaries)

    deltas = [loss for loss in particle.losses if loss.int_type == CONTINUOUS_TYPE]
    assert len(deltas) == 2
    assert sum(loss.e for loss in deltas) == pytest.approx(1e-3)
