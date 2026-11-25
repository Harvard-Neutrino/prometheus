"""Tests for PPC propagator om.conf/om.map setup logic (Step 5).

All tests here mock ``subprocess.Popen`` so they do not require the PPC binary.
Slow integration tests that call the real binary are at the bottom and are
gated behind ``--run-slow``.
"""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prometheus.detector.detector import Detector
from prometheus.detector.medium import Medium
from prometheus.detector.module import Module
from prometheus.photon_propagation.ppc_photon_propagator import ppc_sim

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _legacy_det():
    # Start om_ids at 1: PPC's isinice() requires dom>=1
    mods = [Module(pos=np.array([0.0, 0.0, float(i) * 17.0]), key=(1, i + 1)) for i in range(3)]
    return Detector(mods, Medium.ICE)


def _degg_det():
    # Start om_ids at 1: PPC's isinice() requires dom>=1
    mods = [
        Module(
            pos=np.array([0.0, 0.0, float(i) * 17.0]),
            key=(1, i + 1),
            module_type=120,
            Rr=0.150,
            Rz=0.267,
            n_pmts=2,
            pmt_dirs=[(180.0, 0.0), (0.0, 0.0)],
        )
        for i in range(3)
    ]
    return Detector(mods, Medium.ICE)


def _make_config(tmp_path, om_dirs_path="", ppc_version="official"):
    """Return a minimal ppc_config dict pointing at tmp_path."""
    (tmp_path / "om.wv_120.0").write_text("300 1\n600 1\n")
    return {
        "paths": {
            "ppc_tmpdir": str(tmp_path),
            "ppc_tmpfile": "hits.tmp",
            "f2k_tmpfile": "losses.f2k",
            "ppctables": str(tmp_path),
            "ppc_exe": "/nonexistent/ppc",
            "om_dirs": om_dirs_path,
        },
        "simulation": {
            "device": 0,
            "supress_output": True,
            "ppc_version": ppc_version,
        },
    }


def _dummy_particle():
    """Return a particle-like object that skips the lepton propagation branch."""
    import numpy as np

    # Use a charged pion (211) so ppc_sim creates a point-deposition loss.
    class _FakeParticle:
        pdg_code = 211
        e = 100.0
        position = np.zeros(3)
        direction = np.array([0.0, 0.0, 1.0])
        children = []
        losses = []
        hits = []

        def __int__(self):
            return 211

        def __abs__(self):
            return 211

        def __str__(self):
            return "PiPlus"

    return _FakeParticle()


def _touch_f2k(particle, path, official_ppc=False):
    """Write an empty particle file, which ppc_sim removes after the run."""
    Path(path).touch()


def _touch_geo(path, serial_nos=None):
    """Write an empty geometry file, which ppc_sim removes after the run."""
    Path(path).touch()


class _FakePopen:
    """Minimal Popen stub that writes an empty hit file and returns immediately."""

    def __init__(self, hit_lines=None):
        self._lines = hit_lines or []

    def __call__(self, cmd, shell, stdout, stderr, env):
        # Strip stderr redirect before finding the stdout redirect target.
        # Command form: "ppc N < input > output 2>/dev/null"
        output_path = cmd.split("2>")[0].split(">")[-1].strip().split()[0]
        with open(output_path, "w") as f:
            for line in self._lines:
                f.write(line)
        mock = MagicMock()
        mock.returncode = 0
        mock.communicate = lambda: (b"", b"")
        return mock


# ---------------------------------------------------------------------------
# Setup logic tests (no binary)
# ---------------------------------------------------------------------------


class TestPropagatorSetup:
    def test_no_om_conf_for_legacy_detector(self, tmp_path):
        det = _legacy_det()
        cfg = _make_config(tmp_path)
        particle = _dummy_particle()

        with patch(
            "prometheus.photon_propagation.ppc_photon_propagator.subprocess.Popen", _FakePopen()
        ):
            with patch(
                "prometheus.photon_propagation.ppc_photon_propagator.serialize_to_f2k", _touch_f2k
            ):
                with patch.object(det, "to_f2k", _touch_geo):
                    ppc_sim(particle, det, None, cfg)

        assert not (tmp_path / "om.conf").exists()

    def test_om_conf_written_for_nextgen_detector(self, tmp_path):
        det = _degg_det()
        cfg = _make_config(tmp_path)
        particle = _dummy_particle()

        with patch(
            "prometheus.photon_propagation.ppc_photon_propagator.subprocess.Popen", _FakePopen()
        ):
            with patch(
                "prometheus.photon_propagation.ppc_photon_propagator.serialize_to_f2k", _touch_f2k
            ):
                with patch.object(det, "to_f2k", _touch_geo):
                    ppc_sim(particle, det, None, cfg)

        assert (tmp_path / "om.conf").exists()

    def test_om_map_written_for_nextgen_detector(self, tmp_path):
        det = _degg_det()
        cfg = _make_config(tmp_path)
        particle = _dummy_particle()

        with patch(
            "prometheus.photon_propagation.ppc_photon_propagator.subprocess.Popen", _FakePopen()
        ):
            with patch(
                "prometheus.photon_propagation.ppc_photon_propagator.serialize_to_f2k", _touch_f2k
            ):
                with patch.object(det, "to_f2k", _touch_geo):
                    ppc_sim(particle, det, None, cfg)

        assert (tmp_path / "om.map").exists()

    def test_om_dirs_copied_when_exists(self, tmp_path):
        det = _degg_det()
        src = tmp_path / "om.dirs.src"
        src.write_text("1 0.0 0.0 1.0\n")
        cfg = _make_config(tmp_path, om_dirs_path=str(src))
        particle = _dummy_particle()

        with patch(
            "prometheus.photon_propagation.ppc_photon_propagator.subprocess.Popen", _FakePopen()
        ):
            with patch(
                "prometheus.photon_propagation.ppc_photon_propagator.serialize_to_f2k", _touch_f2k
            ):
                with patch.object(det, "to_f2k", _touch_geo):
                    ppc_sim(particle, det, None, cfg)

        assert (tmp_path / "om.dirs").exists()

    def test_ppctablesdir_env_set(self, tmp_path):
        det = _legacy_det()
        cfg = _make_config(tmp_path)
        particle = _dummy_particle()
        captured_env = {}

        class _CapturePopen:
            def __call__(self, cmd, shell, stdout, stderr, env):
                captured_env.update(env or {})
                output_path = cmd.split(">")[-1].strip().split()[0]
                open(output_path, "w").close()
                mock = MagicMock()
                mock.returncode = 0
                mock.communicate = lambda: (b"", b"")
                return mock

        with patch(
            "prometheus.photon_propagation.ppc_photon_propagator.subprocess.Popen", _CapturePopen()
        ):
            with patch(
                "prometheus.photon_propagation.ppc_photon_propagator.serialize_to_f2k", _touch_f2k
            ):
                with patch.object(det, "to_f2k", _touch_geo):
                    ppc_sim(particle, det, None, cfg)

        assert "PPCTABLESDIR" in captured_env
        assert captured_env["PPCTABLESDIR"] == str(tmp_path)

    def test_legacy_output_pmt_id_none(self, tmp_path):
        det = _legacy_det()
        cfg = _make_config(tmp_path)
        particle = _dummy_particle()
        legacy_hits = ["HIT 1 1 100.0 400.0 1.0 2.0 0.5 1.0\n"]

        with patch(
            "prometheus.photon_propagation.ppc_photon_propagator.subprocess.Popen",
            _FakePopen(legacy_hits),
        ):
            with patch(
                "prometheus.photon_propagation.ppc_photon_propagator.serialize_to_f2k", _touch_f2k
            ):
                with patch.object(det, "to_f2k", _touch_geo):
                    ppc_sim(particle, det, None, cfg)

        assert all(h.pmt_id is None for h in particle.hits)

    def test_nextgen_output_pmt_id_set(self, tmp_path):
        det = _degg_det()
        cfg = _make_config(tmp_path)
        particle = _dummy_particle()
        nextgen_hits = [
            "HIT 1 1_0 100.0 400.0 1.0 2.0 0.5 1.0\n",
            "HIT 1 1_1 200.0 400.0 1.0 2.0 0.5 1.0\n",
        ]

        with patch(
            "prometheus.photon_propagation.ppc_photon_propagator.subprocess.Popen",
            _FakePopen(nextgen_hits),
        ):
            with patch(
                "prometheus.photon_propagation.ppc_photon_propagator.serialize_to_f2k", _touch_f2k
            ):
                with patch.object(det, "to_f2k", _touch_geo):
                    ppc_sim(particle, det, None, cfg)

        pmt_ids = {h.pmt_id for h in particle.hits}
        assert pmt_ids == {0, 1}


def _run_fake_ppc(tmp_path, det, hit_lines, popen=None, **config):
    particle = _dummy_particle()
    module = "prometheus.photon_propagation.ppc_photon_propagator"
    with patch(f"{module}.subprocess.Popen", popen or _FakePopen(hit_lines)):
        with patch(f"{module}.serialize_to_f2k", _touch_f2k):
            with patch.object(det, "to_f2k", _touch_geo):
                ppc_sim(particle, det, None, _make_config(tmp_path, **config))
    return particle


class TestNextgenErrors:
    def test_old_ppc_output_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="ignored om.conf"):
            _run_fake_ppc(tmp_path, _degg_det(), ["HIT 1 1 100.0 400.0 1.0 2.0 0.5 1.0\n"])

    def test_missing_wavelength_table_raises(self, tmp_path):
        det = _degg_det()
        for m in det.modules:
            m.module_type = 999
        with pytest.raises(FileNotFoundError, match="om.wv_999"):
            _run_fake_ppc(tmp_path, det, [])

    def test_inherited_nextgendir_is_replaced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXTGENDIR", "/somewhere/else")
        seen = {}
        fake = _FakePopen()

        def popen(cmd, shell, stdout, stderr, env):
            seen.update(env)
            return fake(cmd, shell, stdout, stderr, env)

        _run_fake_ppc(tmp_path, _degg_det(), [], popen)
        assert seen["NEXTGENDIR"] == str(tmp_path)


class TestPPCVersion:
    _HIT = ["HIT 1 1 100.0 400.0 1.0 2.0 0.5 1.5\n"]

    def test_multi_pmt_needs_official(self, tmp_path):
        with pytest.raises(ValueError, match="need the official PPC"):
            _run_fake_ppc(tmp_path, _degg_det(), [], ppc_version="bundled")

    def test_unknown_version_raises(self, tmp_path):
        with pytest.raises(ValueError, match="ppc_version must be"):
            _run_fake_ppc(tmp_path, _legacy_det(), [], ppc_version="other")

    def test_legacy_angle_order_follows_version(self, tmp_path):
        bundled = _run_fake_ppc(tmp_path, _legacy_det(), self._HIT, ppc_version="bundled").hits[0]
        assert (bundled.om_zenith, bundled.photon_zenith) == (1.0, 0.5)
        official = _run_fake_ppc(tmp_path, _legacy_det(), self._HIT).hits[0]
        assert (official.om_zenith, official.photon_zenith) == (0.5, 1.0)
        assert (official.om_azimuth, official.photon_azimuth) == (1.5, 2.0)

    def test_legacy_track_angles_follow_version(self, tmp_path):
        module = "prometheus.photon_propagation.ppc_photon_propagator"
        seen = []

        def write(particle, path, official_ppc=False):
            seen.append(official_ppc)
            _touch_f2k(particle, path)

        for version in ("bundled", "official"):
            det = _legacy_det()
            with patch(f"{module}.subprocess.Popen", _FakePopen()):
                with patch(f"{module}.serialize_to_f2k", write):
                    with patch.object(det, "to_f2k", _touch_geo):
                        cfg = _make_config(tmp_path, ppc_version=version)
                        ppc_sim(_dummy_particle(), det, None, cfg)
        assert seen == [False, True]


class TestModuleFiles:
    def test_om_dirs_generated(self, tmp_path):
        _run_fake_ppc(tmp_path, _degg_det(), [])
        directions = np.loadtxt(tmp_path / "om.dirs")[:, 1:]
        assert len(directions) == 642
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-8)
        np.testing.assert_allclose(directions.mean(axis=0), 0, atol=1e-8)

    def test_missing_om_dirs_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="om_dirs file"):
            _run_fake_ppc(tmp_path, _degg_det(), [], om_dirs_path=str(tmp_path / "absent"))

    def test_om_tables_directory_is_used(self, tmp_path):
        tables = tmp_path / "tables"
        tables.mkdir()
        (tables / "om.wv_121.0").write_text("300 1\n600 1\n")
        work = tmp_path / "work"
        work.mkdir()
        det = _degg_det()
        for m in det.modules:
            m.module_type = 121
        cfg = _make_config(work)
        cfg["paths"]["om_tables"] = str(tables)
        module = "prometheus.photon_propagation.ppc_photon_propagator"
        with patch(f"{module}.subprocess.Popen", _FakePopen()):
            with patch(f"{module}.serialize_to_f2k", _touch_f2k):
                with patch.object(det, "to_f2k", _touch_geo):
                    ppc_sim(_dummy_particle(), det, None, cfg)
        assert (work / "om.wv_121.0").exists()


# ---------------------------------------------------------------------------
# Slow integration tests (require real PPC binary)
# ---------------------------------------------------------------------------


def _ppc_exe():
    """Return PPC executable path from env var or skip the test."""
    exe = os.environ.get("PPC_EXE", "")
    if not exe or not os.path.isfile(exe):
        pytest.skip("Set PPC_EXE env var to the ppc binary to run this test")
    return exe


def _ppc_tables():
    exe = os.environ.get("PPC_TABLES_DIR", "")
    if not exe or not os.path.isdir(exe):
        pytest.skip("Set PPC_TABLES_DIR env var to the PPC tables directory to run this test")
    return exe


@pytest.mark.slow
class TestPPCIntegration:
    def _run_ppc(self, tmp_path, det, tables_dir, ppc_exe, om_dirs=""):
        """Run a minimal PPC simulation and return hits."""
        from prometheus.photon_propagation.ppc_photon_propagator import ppc_sim

        # Copy tables
        sim_dir = tmp_path / "tables"
        shutil.copytree(tables_dir, str(sim_dir))

        cfg = {
            "paths": {
                "ppc_tmpdir": str(sim_dir),
                "ppc_tmpfile": "hits.tmp",
                "f2k_tmpfile": "losses.f2k",
                "ppctables": tables_dir,
                "ppc_exe": ppc_exe,
                "om_dirs": om_dirs,
            },
            "simulation": {
                "device": 0,
                "supress_output": True,
                "ppc_version": os.environ.get("PPC_VERSION", "official"),
            },
        }

        class _Particle:
            pdg_code = 211
            e = 1000.0
            time = 0.0
            position = np.zeros(3)
            direction = np.array([0.0, 0.0, 1.0])
            children = []
            hits = []
            losses = []

            def __int__(self):
                return 211

            def __abs__(self):
                return 211

            def __str__(self):
                return "PiPlus"

        p = _Particle()
        from prometheus.lepton_propagation.loss import Loss

        p.losses = [Loss(211, 1000.0, np.zeros(3))]
        ppc_sim(p, det, None, cfg)
        return p.hits

    def test_legacy_detector_returns_hits(self, tmp_path):
        exe = _ppc_exe()
        tables = _ppc_tables()
        det = _legacy_det()
        hits = self._run_ppc(tmp_path, det, tables, exe)
        assert len(hits) > 0

    def test_legacy_hits_pmt_id_none(self, tmp_path):
        exe = _ppc_exe()
        tables = _ppc_tables()
        det = _legacy_det()
        hits = self._run_ppc(tmp_path, det, tables, exe)
        assert all(h.pmt_id is None for h in hits)

    def test_degg_hits_have_pmt_indices(self, tmp_path):
        exe = _ppc_exe()
        tables = _ppc_tables()
        om_dirs = os.environ.get("PPC_OM_DIRS", "")
        det = _degg_det()
        hits = self._run_ppc(tmp_path, det, tables, exe, om_dirs=om_dirs)
        assert len(hits) > 0
        assert all(h.pmt_id in (0, 1) for h in hits)

    def test_degg_both_pmt_indices_appear(self, tmp_path):
        exe = _ppc_exe()
        tables = _ppc_tables()
        om_dirs = os.environ.get("PPC_OM_DIRS", "")
        det = _degg_det()
        hits = self._run_ppc(tmp_path, det, tables, exe, om_dirs=om_dirs)
        pmt_ids = {h.pmt_id for h in hits}
        assert 0 in pmt_ids
        assert 1 in pmt_ids
