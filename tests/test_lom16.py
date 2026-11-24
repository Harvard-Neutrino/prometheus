"""Tests of the LOM-16 shipped in resources/optical_modules.

The slow tests need the official PPC in PPC_EXE and ice tables in PPC_TABLES_DIR.
"""

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from prometheus.detector import Detector, Medium, Module, read_om_conf

_ROOT = Path(__file__).parents[1]
_LOM16 = _ROOT / "resources" / "optical_modules" / "lom16"
_LI_FILE = _ROOT / "tests" / "resources" / "MuMinus_Hadrons_seed_925_LI_output.h5"
_CENTER = np.array([0.0, 0.0, -2150.0])
_N_PMTS = 16


def _detector():
    kwargs = read_om_conf(_LOM16 / "om.conf")[171]
    modules = []
    for string, (x, y) in enumerate([(30.0, 0.0), (-30.0, 0.0), (0.0, 30.0), (0.0, -30.0)]):
        for i in range(10):
            position = np.array([x, y, _CENTER[2] + 17.0 * (i - 4.5)])
            modules.append(Module(position, (string + 1, i + 1), **kwargs))
    return Detector(modules, Medium.ICE)


def _ppc():
    exe = os.environ.get("PPC_EXE", "")
    tables = os.environ.get("PPC_TABLES_DIR", "")
    if not os.path.isfile(exe) or not os.path.isdir(tables):
        pytest.skip("Set PPC_EXE (official PPC) and PPC_TABLES_DIR")
    return exe, tables


def _pmt_axes():
    zenith, azimuth = np.radians(np.array(read_om_conf(_LOM16 / "om.conf")[171]["pmt_dirs"])).T
    return np.stack(
        [np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)], -1
    )


def test_shipped_files_match_the_fit(tmp_path):
    subprocess.run(
        [sys.executable, str(_ROOT / "examples" / "06_fit_lom_shape.py"), "--out-dir", tmp_path],
        check=True,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
        capture_output=True,
    )
    for name in ("om.conf", "om.wv_171.0"):
        assert (tmp_path / name).read_text() == (_LOM16 / name).read_text()


def _write_events(path, directions):
    """Write LeptonInjector events at the center, with the given zenith angles."""
    n = len(directions)
    with h5py.File(_LI_FILE) as source, h5py.File(path, "w") as out:
        name = next(iter(source))
        for key, dataset in source[name].items():
            rows = dataset[:n]
            if key == "properties":
                for i, coordinate in enumerate("xyz"):
                    rows[coordinate] = _CENTER[i]
                rows["totalEnergy"] = 1000.0
                rows["zenith"] = directions
            else:
                rows["Position"] = _CENTER
                rows["Direction"][:, 0] = directions
                rows["Direction"][:, 1] = 0.0
                rows["Energy"] = {"initial": 1000.0, "final_1": 800.0, "final_2": 200.0}[key]
            out.create_dataset(f"{name}/{key}", data=rows)


@pytest.mark.slow
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_cascade_lights_the_side_it_points_to(tmp_path, sign):
    from prometheus.particle import PropagatableParticle
    from prometheus.photon_propagation.ppc_photon_propagator import ppc_sim

    exe, tables = _ppc()
    work = tmp_path / "work"
    shutil.copytree(tables, work, ignore=shutil.ignore_patterns("geo-f2k", "eff-f2k", "dx.dat"))
    particle = PropagatableParticle(211, 2000.0, _CENTER, np.array([0.0, 0.0, sign]), 0, None, 0.0)
    config = {
        "paths": {
            "ppc_tmpdir": str(work),
            "ppc_tmpfile": "hits",
            "f2k_tmpfile": "losses",
            "ppctables": tables,
            "ppc_exe": exe,
            "om_tables": str(_LOM16),
        },
        "simulation": {"device": 0, "supress_output": True, "ppc_version": "official"},
    }
    detector = _detector()
    ppc_sim(particle, detector, None, config)
    z = np.array([detector[(h.string_id, h.om_id)].pos[2] for h in particle.hits]) - _CENTER[2]
    assert len(z) > 100
    assert (z * sign > 0).mean() > 0.6


@pytest.mark.slow
def test_full_chain(tmp_path):
    import pyarrow.parquet as pq

    pytest.importorskip("proposal")
    from prometheus import Prometheus
    from prometheus import config as base

    exe, tables = _ppc()
    ice = tmp_path / "ice"
    shutil.copytree(tables, ice, ignore=shutil.ignore_patterns("geo-f2k", "eff-f2k", "dx.dat"))
    events = tmp_path / "events.h5"
    _write_events(events, [0.0, np.pi])

    cfg = copy.deepcopy(base)
    cfg.run.run_number = 1
    cfg.run.random_state_seed = 1
    cfg.run.nevents = 2
    cfg.run.storage_prefix = str(tmp_path) + "/"
    cfg.injection.name = "LeptonInjector"
    cfg.injection.lepton_injector.inject = False
    cfg.injection.lepton_injector.paths.injection_file = str(events)
    cfg.photon_propagator.name = "PPC"
    ppc = cfg.photon_propagator.ppc
    ppc.paths.ppc_exe = exe
    ppc.paths.ppctables = str(ice)
    ppc.paths.om_tables = str(_LOM16)
    ppc.simulation.ppc_version = "official"
    ppc.simulation.output_mode = "extended"
    Prometheus(cfg, detector=_detector()).sim()

    (path,) = tmp_path.glob("*.parquet")
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 2
    axes = _pmt_axes()
    for row, sign in zip(rows, (1.0, -1.0)):
        photons = {key: np.array(value) for key, value in row["photons"].items()}
        z = photons["sensor_pos_z"] - _CENTER[2]
        assert (z * sign > 0).mean() > 0.6

        assert set(photons["pmt_id"]) <= set(range(_N_PMTS))
        zenith, azimuth = photons["photon_zenith"], photons["photon_azimuth"]
        travel = np.stack(
            [np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)],
            -1,
        )
        # A nearly flat PMT sees almost nothing from behind.
        facing = -(travel * axes[photons["pmt_id"]]).sum(axis=1)
        assert (facing > 0).mean() > 0.95

        radius = np.hypot(photons["hit_x"], photons["hit_y"]) / 0.156
        surface = radius**2 + (photons["hit_z"] / 0.222) ** 2
        np.testing.assert_allclose(surface, 1, atol=1e-3)
