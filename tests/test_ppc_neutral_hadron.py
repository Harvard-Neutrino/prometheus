"""Issue #2: ppc_sim must not drop neutral-hadron (pi0/K0) decay-product light.

These tests need no PPC binary/GPU. The deposit tests monkeypatch
`should_propagate` to False so ppc_sim returns right after depositing the Loss
(the should_propagate gate sits after the deposit and before the PPC
subprocess), letting us inspect particle.losses directly.
"""

import numpy as np
import pytest

import prometheus.photon_propagation.ppc_photon_propagator as ppc_mod
from prometheus.lepton_propagation import Loss
from prometheus.particle import PropagatableParticle
from prometheus.utils import serialize_to_f2k
from prometheus.utils.translators import int_type_to_str


class _FakeDetector:
    """Minimal stand-in exposing only what ppc_sim's dispatch reads."""

    offset = np.zeros(3)
    outer_radius = 100.0  # r_inice = outer_radius + 1000 = 1100 m


def _min_config():
    # Only the keys ppc_sim touches before the should_propagate return.
    return {
        "paths": {
            "ppc_tmpdir": "/tmp",
            "ppc_tmpfile": "ppc",
            "f2k_tmpfile": "f2k",
            "ppc_exe": "ppc",
        },
        "simulation": {"device": "cpu", "supress_output": True},
    }


def _run_dispatch(pdg, monkeypatch):
    """Run ppc_sim far enough to exercise the deposit, then bail before PPC."""
    monkeypatch.setattr(ppc_mod, "should_propagate", lambda p: False)
    particle = PropagatableParticle(pdg, 1000.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0, None, 0.0)
    # lp=None is safe: only charged leptons (11/13/15) touch the propagator.
    ppc_mod.ppc_sim(particle, _FakeDetector(), None, _min_config())
    return particle


def test_pi0_deposits_em_cascade(monkeypatch):
    particle = _run_dispatch(111, monkeypatch)
    assert len(particle.losses) == 1
    assert str(particle.losses[0]) == "epair"  # EM: pi0 -> gamma gamma


def test_k0_deposits_hadronic_cascade(monkeypatch):
    particle = _run_dispatch(311, monkeypatch)
    assert len(particle.losses) == 1
    assert str(particle.losses[0]) == "hadr"


def test_antik0_deposits_like_k0(monkeypatch):
    # anti-K0 (-311) must behave identically to K0 (+311): one hadronic Loss.
    # _run_dispatch also exercises str(particle) (tmpfile naming), so this
    # implicitly checks PDG_to_pstring resolves -311 (no KeyError).
    particle = _run_dispatch(-311, monkeypatch)
    assert len(particle.losses) == 1
    assert str(particle.losses[0]) == "hadr"


def test_antipi0_falls_through_to_raise(monkeypatch):
    # pi0 is its own antiparticle; a nonsense -111 must NOT be silently
    # deposited. It should hit the else: raise ValueError, not a KeyError.
    monkeypatch.setattr(ppc_mod, "should_propagate", lambda p: False)
    p = PropagatableParticle(-111, 1000.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0, None, 0.0)
    with pytest.raises(ValueError):
        ppc_mod.ppc_sim(p, _FakeDetector(), None, _min_config())


def test_neutral_hadrons_do_not_raise(monkeypatch):
    # Regression: pre-fix these returned; a future refactor must not route
    # them to the else: raise ValueError branch.
    for pdg in (111, 311):
        _run_dispatch(pdg, monkeypatch)  # must not raise


def test_serialize_emits_correct_tr_names(tmp_path):
    for pdg, expected in ((111, "epair"), (311, "hadr")):
        p = PropagatableParticle(pdg, 500.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0, None, 0.0)
        p.losses.append(Loss(pdg, 500.0, np.zeros(3)))
        fname = tmp_path / f"out_{pdg}.f2k"
        serialize_to_f2k(p, str(fname))
        tr = [ln for ln in fname.read_text().splitlines() if ln.startswith("TR")]
        assert len(tr) == 1
        # TR 0 0 <name> x y z ...  -> name is token index 3
        assert tr[0].split()[3] == expected


def test_cascade_mapping_invariants():
    # The fix relies on these; guard against silent dict edits.
    assert int_type_to_str[111] == "epair"
    assert int_type_to_str[311] == "hadr"
