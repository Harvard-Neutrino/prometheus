"""Tests specific to non-spherical and multi-PMT modules (new-types chapter).

Fast tests cover geometry correctness and config round-trips without needing
the PPC binary.  Slow integration tests are gated behind --run-slow and require
environment variables PPC_EXE and PPC_TABLES_DIR to be set.
"""

import os

import numpy as np
import pytest

from prometheus.detector.detector import Detector
from prometheus.detector.medium import Medium
from prometheus.detector.module import Module
from prometheus.photon_propagation.hit import Hit
from prometheus.photon_propagation.utils.parse_ppc import parse_ppc
from prometheus.utils.serialization.serialize_particles_to_awkward import (
    _hit_cartesian,
    serialize_particles_to_awkward,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _degg_mod(key=(1, 1), z=0.0):
    return Module(
        pos=np.array([0.0, 0.0, z]),
        key=key,
        module_type=120,
        Rr=0.150,
        Rz=0.267,
        beta=0.5,
        area=1.0,
        n_pmts=2,
        pmt_dirs=[(180.0, 0.0), (0.0, 0.0)],
    )


def _wom_mod(key=(1, 2), z=10.0):
    return Module(
        pos=np.array([0.0, 0.0, z]),
        key=key,
        module_type=200,
        Rr=0.06,
        Rz=-0.38,
        beta=-2.0,
        area=1.0,
        n_pmts=2,
        pmt_dirs=[(180.0, 0.0), (0.0, 0.0)],
    )


def _parse_om_conf_types(path):
    types = {}
    current = None
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if not line[0].isspace():
                tokens = line.split()
                current = int(tokens[1])
                types[current] = {
                    "Rr": float(tokens[4]),
                    "Rz": float(tokens[5]),
                    "n_pmts": int(tokens[6]),
                    "pmt_dirs": [(float(tokens[7]), float(tokens[8]))],
                }
            else:
                stripped = line.strip().split()
                if stripped and current is not None:
                    types[current]["pmt_dirs"].append((float(stripped[0]), float(stripped[1])))
    return types


def _write_ppc_file(tmp_path, lines):
    p = tmp_path / "ppc_out.txt"
    p.write_text("".join(lines))
    return str(p)


# ---------------------------------------------------------------------------
# om.conf geometry correctness (no binary)
# ---------------------------------------------------------------------------


class TestOmConfGeometry:
    def test_spheroid_aspect_ratio(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf_types(path)
        F = types[120]["Rz"] / types[120]["Rr"]
        assert F == pytest.approx(0.267 / 0.150, rel=1e-5)

    def test_cylinder_rz_negative(self, tmp_path):
        mods = [_wom_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf_types(path)
        assert types[200]["Rz"] < 0

    def test_degg_two_pmt_dir_lines(self, tmp_path):
        mods = [_degg_mod()]
        det = Detector(mods, Medium.ICE)
        path = str(tmp_path / "om.conf")
        det.to_om_conf(path)
        types = _parse_om_conf_types(path)
        assert len(types[120]["pmt_dirs"]) == 2

    def test_om_map_type_ids_subset_of_conf(self, tmp_path):
        mods = [_degg_mod(key=(1, 0)), _wom_mod(key=(1, 1))]
        det = Detector(mods, Medium.ICE)
        det.to_om_conf(str(tmp_path / "om.conf"))
        det.to_om_map(str(tmp_path / "om.map"))
        conf_ids = set(_parse_om_conf_types(str(tmp_path / "om.conf")).keys())
        with open(str(tmp_path / "om.map")) as f:
            map_ids = {int(line.split()[2]) for line in f if line.strip()}
        assert map_ids.issubset(conf_ids)


# ---------------------------------------------------------------------------
# Hit position geometry (no binary)
# ---------------------------------------------------------------------------


def _mod(Rr, Rz):
    return Module(pos=np.zeros(3), key=(1, 1), Rr=Rr, Rz=Rz)


class TestHitCartesianGeometry:
    def test_sphere_surface(self):
        R = 0.16510
        mod = _mod(R, R)
        for th in np.linspace(0.1, np.pi - 0.1, 7):
            for ph in np.linspace(0, 2 * np.pi, 5, endpoint=False):
                h = Hit(1, 1, 0.0, 400.0, th, ph, 0.0, 0.0)
                hx, hy, hz = _hit_cartesian(h, mod)
                assert np.sqrt(hx**2 + hy**2 + hz**2) == pytest.approx(R, rel=1e-5)

    def test_spheroid_north_pole(self):
        Rr, Rz = 0.150, 0.267
        mod = _mod(Rr, Rz)
        h = Hit(1, 1, 0.0, 400.0, 0.0, 0.0, 0.0, 0.0)
        hx, hy, hz = _hit_cartesian(h, mod)
        assert hx == pytest.approx(0.0, abs=1e-10)
        assert hy == pytest.approx(0.0, abs=1e-10)
        assert abs(hz) == pytest.approx(Rz, rel=1e-5)

    def test_spheroid_equator(self):
        Rr, Rz = 0.150, 0.267
        mod = _mod(Rr, Rz)
        h = Hit(1, 1, 0.0, 400.0, np.pi / 2, 0.0, 0.0, 0.0)
        hx, hy, hz = _hit_cartesian(h, mod)
        assert hz == pytest.approx(0.0, abs=1e-10)
        assert np.sqrt(hx**2 + hy**2) == pytest.approx(Rr, rel=1e-5)

    def test_spheroid_on_surface(self):
        Rr, Rz = 0.150, 0.267
        mod = _mod(Rr, Rz)
        for th in np.linspace(0.1, np.pi - 0.1, 7):
            for ph in np.linspace(0, 2 * np.pi, 5, endpoint=False):
                h = Hit(1, 1, 0.0, 400.0, th, ph, 0.0, 0.0)
                hx, hy, hz = _hit_cartesian(h, mod)
                on_surf = (hx / Rr) ** 2 + (hy / Rr) ** 2 + (hz / Rz) ** 2
                assert on_surf == pytest.approx(1.0, rel=1e-5)

    def test_cylinder_wall_radius(self):
        Rr, Rz = 0.06, -0.38
        mod = _mod(Rr, Rz)
        for ph in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            h = Hit(1, 1, 0.0, 400.0, np.pi / 3, ph, 0.0, 0.0)
            hx, hy, hz = _hit_cartesian(h, mod)
            assert np.sqrt(hx**2 + hy**2) == pytest.approx(Rr, rel=1e-5)

    def test_cylinder_z_range(self):
        Rr, Rz = 0.06, -0.38
        mod = _mod(Rr, Rz)
        for th in np.linspace(0.01, np.pi - 0.01, 9):
            h = Hit(1, 1, 0.0, 400.0, th, 0.5, 0.0, 0.0)
            _, _, hz = _hit_cartesian(h, mod)
            assert abs(hz) <= abs(Rz) + 1e-9


# ---------------------------------------------------------------------------
# PMT index parsing (no binary)
# ---------------------------------------------------------------------------


class TestPMTIndexParsing:
    def test_pmt0_and_pmt1_from_nextgen_file(self, tmp_path):
        lines = [
            "HIT 1 1_0 100.0 400.0 1.0 2.0 0.5 1.0\n",
            "HIT 1 1_1 200.0 400.0 1.0 2.0 0.5 1.0\n",
        ]
        hits = parse_ppc(_write_ppc_file(tmp_path, lines))
        ids = {h.pmt_id for h in hits}
        assert ids == {0, 1}

    def test_pmt_id_in_photon_output_mode(self, tmp_path):
        import awkward as ak

        class _FakeMod:
            key = (1, 1)
            pos = np.zeros(3)
            Rr = 0.16510
            Rz = 0.16510
            module_type = 120

        class _FakeDet:
            def __getitem__(self, k):
                return _FakeMod()

        class _FakeP:
            hits = [
                Hit(1, 1, 100.0, 400.0, 1.0, 0.5, 0.8, 1.2, pmt_id=0),
                Hit(1, 1, 200.0, 400.0, 1.0, 0.5, 0.8, 1.2, pmt_id=1),
            ]
            serialization_idx = 0
            children = []

        class _FakeEv:
            final_states = [_FakeP()]

        class _FakeInj:
            def __iter__(self):
                return iter([_FakeEv()])

        result = serialize_particles_to_awkward(_FakeDet(), _FakeInj(), output_mode="photon")
        pmt_ids = ak.to_list(result["pmt_id"])[0]
        assert set(pmt_ids) == {0, 1}


# ---------------------------------------------------------------------------
# Slow integration tests (real PPC binary)
# ---------------------------------------------------------------------------


def _require_ppc():
    exe = os.environ.get("PPC_EXE", "")
    tables = os.environ.get("PPC_TABLES_DIR", "")
    if not exe or not os.path.isfile(exe):
        pytest.skip("Set PPC_EXE to run slow PPC integration tests")
    if not tables or not os.path.isdir(tables):
        pytest.skip("Set PPC_TABLES_DIR to run slow PPC integration tests")
    return exe, tables


def _run_minimal_ppc(tmp_path, det, exe, tables, om_dirs=""):
    """Run PPC with a bright point source and return hits."""
    import shutil

    from prometheus.lepton_propagation.loss import Loss
    from prometheus.photon_propagation.ppc_photon_propagator import ppc_sim

    sim_dir = tmp_path / "sim"
    shutil.copytree(tables, str(sim_dir))

    cfg = {
        "paths": {
            "ppc_tmpdir": str(sim_dir),
            "ppc_tmpfile": "hits.tmp",
            "f2k_tmpfile": "losses.f2k",
            "ppctables": tables,
            "ppc_exe": exe,
            "om_dirs": om_dirs,
        },
        "simulation": {
            "device": 0,
            "supress_output": True,
            "ppc_version": os.environ.get("PPC_VERSION", "official"),
        },
    }

    class _P:
        pdg_code = 211
        e = 5000.0
        time = 0.0
        position = np.zeros(3)
        direction = np.array([0.0, 0.0, 1.0])
        children = []
        hits = []
        losses = [Loss(211, 5000.0, np.zeros(3))]

        def __int__(self):
            return 211

        def __abs__(self):
            return 211

        def __str__(self):
            return "PiPlus"

    p = _P()
    ppc_sim(p, det, None, cfg)
    return p.hits


@pytest.mark.slow
class TestPPCNonSphericalIntegration:
    def test_spherical_hits_pmt_id_none(self, tmp_path):
        exe, tables = _require_ppc()
        mods = [Module(pos=np.array([0.0, 0.0, float(i) * 17.0]), key=(1, i + 1)) for i in range(5)]
        det = Detector(mods, Medium.ICE)
        hits = _run_minimal_ppc(tmp_path, det, exe, tables)
        assert len(hits) > 0
        assert all(h.pmt_id is None for h in hits)

    def test_degg_hits_pmt_indices_0_and_1(self, tmp_path):
        exe, tables = _require_ppc()
        om_dirs = os.environ.get("PPC_OM_DIRS", "")
        mods = [_degg_mod(key=(1, i + 1), z=float(i) * 17.0) for i in range(5)]
        det = Detector(mods, Medium.ICE)
        hits = _run_minimal_ppc(tmp_path, det, exe, tables, om_dirs=om_dirs)
        assert len(hits) > 0
        pmt_ids = {h.pmt_id for h in hits}
        assert pmt_ids.issubset({0, 1})
        assert 0 in pmt_ids
        assert 1 in pmt_ids

    def test_degg_hits_lie_on_spheroid_surface(self, tmp_path):
        exe, tables = _require_ppc()
        om_dirs = os.environ.get("PPC_OM_DIRS", "")
        mods = [_degg_mod(key=(1, i + 1), z=float(i) * 17.0) for i in range(5)]
        det = Detector(mods, Medium.ICE)
        hits = _run_minimal_ppc(tmp_path, det, exe, tables, om_dirs=om_dirs)
        assert len(hits) > 0
        Rr, Rz = 0.150, 0.267
        mod = _degg_mod()
        for h in hits:
            if h.om_zenith is None or h.om_azimuth is None:
                continue
            hx, hy, hz = _hit_cartesian(h, mod)
            on_surf = (hx / Rr) ** 2 + (hy / Rr) ** 2 + (hz / Rz) ** 2
            # Tolerance is relaxed due to oversize-factor geometry correction
            assert on_surf == pytest.approx(1.0, abs=0.05)

    def test_wom_hits_pmt_indices_0_and_1(self, tmp_path):
        exe, tables = _require_ppc()
        om_dirs = os.environ.get("PPC_OM_DIRS", "")
        mods = [_wom_mod(key=(1, i + 1), z=float(i) * 17.0) for i in range(5)]
        det = Detector(mods, Medium.ICE)
        hits = _run_minimal_ppc(tmp_path, det, exe, tables, om_dirs=om_dirs)
        assert len(hits) > 0
        pmt_ids = {h.pmt_id for h in hits}
        assert pmt_ids.issubset({0, 1})

    def test_full_output_round_trips_to_parquet(self, tmp_path):
        import awkward as ak

        exe, tables = _require_ppc()
        om_dirs = os.environ.get("PPC_OM_DIRS", "")
        mods = [_degg_mod(key=(1, i + 1), z=float(i) * 17.0) for i in range(5)]
        det = Detector(mods, Medium.ICE)
        hits = _run_minimal_ppc(tmp_path, det, exe, tables, om_dirs=om_dirs)
        assert len(hits) > 0

        # Build a minimal injection-like structure from the raw hits
        class _P:
            serialization_idx = 0
            children = []

            def __init__(self, hs):
                self.hits = hs

        class _Ev:
            def __init__(self, hs):
                self.final_states = [_P(hs)]

        class _Inj:
            def __init__(self, hs):
                self._ev = [_Ev(hs)]

            def __iter__(self):
                return iter(self._ev)

        result = serialize_particles_to_awkward(det, _Inj(hits), output_mode="full")
        assert result is not None

        parquet_path = str(tmp_path / "out.parquet")
        ak.to_parquet(result, parquet_path)
        reloaded = ak.from_parquet(parquet_path)

        assert "hit_x" in reloaded.fields
        assert "hit_y" in reloaded.fields
        assert "hit_z" in reloaded.fields
        assert "pmt_id" in reloaded.fields

        # All values should be finite (no NaN/None from valid hits)
        hit_x_vals = ak.to_numpy(ak.flatten(reloaded["hit_x"]))
        assert np.all(np.isfinite(hit_x_vals))
