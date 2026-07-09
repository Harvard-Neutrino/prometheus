"""Issue #4: decay-product light must be timed from the decay vertex.

A daughter particle starts at time ``L_decay / c`` (the parent's flight time to
the decay vertex), so its loss/hit times must carry that offset instead of
restarting the clock at 0. These tests need no PPC binary/GPU.
"""

import numpy as np
import pytest

from prometheus.lepton_propagation import Loss
from prometheus.particle import PropagatableParticle, particle_from_proposal
from prometheus.utils import serialize_to_f2k
from prometheus.utils.units import SpeedOfLight, s_to_ns

_C_M_PER_NS = SpeedOfLight / s_to_ns  # m/ns
_DIR_Z = np.array([0.0, 0.0, 1.0])


def _tr_times(path):
    return [float(ln.split()[11]) for ln in path.read_text().splitlines() if ln.startswith("TR")]


def _particle(time=0.0, pdg=15):
    return PropagatableParticle(pdg, 1e3, np.zeros(3), _DIR_Z, 0, None, time=time)


class TestParticleTimeField:
    def test_time_is_required(self):
        # `time` has no default: omitting it must fail loud, not silently
        # restart the clock at 0 (the Issue #4 bug class).
        with pytest.raises(TypeError):
            PropagatableParticle(15, 1e3, np.zeros(3), _DIR_Z, 0, None)

    def test_time_is_stored(self):
        assert _particle(time=1633.0).time == pytest.approx(1633.0)


class TestLossTiming:
    def test_primary_loss_timed_from_zero(self, tmp_path):
        d = 300.0
        p = _particle(time=0.0)
        p.losses.append(Loss(-2000001006, 500.0, np.array([0.0, 0.0, d])))
        out = tmp_path / "primary.f2k"
        serialize_to_f2k(p, str(out))
        assert _tr_times(out)[0] == pytest.approx(d / _C_M_PER_NS)

    def test_daughter_loss_includes_flight_offset(self, tmp_path):
        d = 300.0
        offset = 1633.0  # ns, ~490 m tau flight
        p = _particle(time=offset)
        p.losses.append(Loss(-2000001006, 500.0, np.array([0.0, 0.0, d])))
        out = tmp_path / "daughter.f2k"
        serialize_to_f2k(p, str(out))
        assert _tr_times(out)[0] == pytest.approx(offset + d / _C_M_PER_NS)

    def test_child_serialized_with_its_own_time(self, tmp_path):
        # serialize_to_f2k also writes the parent's children; each child's loss
        # must be timed from that child's start time.
        offset = 1633.0
        parent = _particle(time=0.0)
        child = PropagatableParticle(11, 500.0, np.zeros(3), _DIR_Z, 0, parent, time=offset)
        child.losses.append(Loss(-2000001006, 500.0, np.array([0.0, 0.0, 150.0])))
        parent.children.append(child)
        out = tmp_path / "family.f2k"
        serialize_to_f2k(parent, str(out))
        assert _tr_times(out)[0] == pytest.approx(offset + 150.0 / _C_M_PER_NS)


class _V:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _PPParticle:
    """Duck-typed PROPOSAL particle (energy in MeV, position in cm, time in s)."""

    type = 11
    energy = 500_000.0
    position = _V(0.0, 0.0, 49000.0)  # 490 m
    direction = _V(0.0, 0.0, 1.0)
    time = 1.634e-6  # s


class TestParticleFromProposal:
    def test_time_converted_seconds_to_ns(self):
        child = particle_from_proposal(_PPParticle(), np.zeros(3), parent=_particle())
        assert child.time == pytest.approx(_PPParticle.time * s_to_ns)

    def test_zero_time_stays_zero(self):
        pp = _PPParticle()
        pp.time = 0.0
        child = particle_from_proposal(pp, np.zeros(3), parent=_particle())
        assert child.time == 0.0
