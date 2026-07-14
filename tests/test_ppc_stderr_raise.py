"""ppc_sim must fail loudly when the PPC subprocess exits nonzero.

Previously stderr was redirected to /dev/null and the return code was ignored,
so a bad exe path / missing tables / crash looked identical to "no photon hits".
These tests drive ppc_sim to the subprocess call with a fake Popen (no PPC
binary, no GPU) and assert the exit code is now honored.
"""

import logging

import numpy as np
import pytest

import prometheus.photon_propagation.ppc_photon_propagator as ppc_mod
from prometheus.particle import PropagatableParticle


class _FakeDetector:
    """Minimal stand-in exposing only what ppc_sim reads before the subprocess."""

    offset = np.zeros(3)
    outer_radius = 100.0  # r_inice = outer_radius + 1000 = 1100 m
    modules = ()  # -> serial_nos == []

    def to_f2k(self, path, serial_nos=None):
        pass


class _FakeProc:
    """Stand-in for subprocess.Popen's return value."""

    def __init__(self, returncode, stderr=b""):
        self.returncode = returncode
        self._stderr = stderr

    def communicate(self):
        return b"", self._stderr


def _config():
    return {
        "paths": {
            "ppc_tmpdir": "/tmp",
            "ppc_tmpfile": "ppc",
            "f2k_tmpfile": "f2k",
            "ppc_exe": "ppc",
        },
        "simulation": {"device": "cpu", "supress_output": False},
    }


def _drive(monkeypatch, returncode, stderr=b""):
    """Run ppc_sim up to and through the (faked) PPC subprocess call."""
    monkeypatch.setattr(ppc_mod, "should_propagate", lambda p: True)
    monkeypatch.setattr(ppc_mod, "serialize_to_f2k", lambda *a, **k: None)
    monkeypatch.setattr(ppc_mod.os, "remove", lambda *a, **k: None)
    monkeypatch.setattr(ppc_mod, "parse_ppc", lambda f: ["HIT"])
    monkeypatch.setattr(ppc_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(returncode, stderr))
    # Hadron (2212): lp=None is safe; deposits a loss then reaches the subprocess.
    particle = PropagatableParticle(
        2212, 1000.0, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0, None, 0.0
    )
    ppc_mod.ppc_sim(particle, _FakeDetector(), None, _config())
    return particle


def test_nonzero_exit_raises_with_stderr(monkeypatch, caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="exit code 3"):
            _drive(monkeypatch, returncode=3, stderr=b"PPC: cannot open tables\n")
    # The real stderr is surfaced in the logs, not swallowed.
    assert "cannot open tables" in caplog.text


def test_zero_exit_does_not_raise_and_parses_hits(monkeypatch):
    particle = _drive(monkeypatch, returncode=0, stderr=b"")
    assert particle.hits == ["HIT"]


def test_records_status_in_subprocess_statuses(monkeypatch):
    before = len(ppc_mod.subprocess_statuses)
    _drive(monkeypatch, returncode=0, stderr=b"")
    assert ppc_mod.subprocess_statuses[-1]["returncode"] == 0
    assert len(ppc_mod.subprocess_statuses) == before + 1
