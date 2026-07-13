from types import SimpleNamespace

import numpy as np
import pytest

from prometheus.injection.injection.genie_injection import (
    _random_rotation,
    injection_from_genie_output,
)
from prometheus.injection.injection.LI_injection import injection_from_LI_output

GENIE_FILE = "tests/resources/genie_example.root"


def _genie_sim_config(**overrides):
    base = {
        "placement": "fixed",
        "positions": None,
        "n_events": None,
        "random_state_seed": 7,
        "interaction_filter": None,
        "direction_mode": "as-is",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_li_injection_loading() -> None:
    injection = injection_from_LI_output("tests/resources/MuMinus_Hadrons_seed_925_LI_output.h5")
    assert len(injection) == 50
    assert all([len(event.final_states) == 2 for event in injection])
    assert all([event.interaction.name == "CHARGED_CURRENT" for event in injection])
    assert all([event.initial_state.pdg_code == 14 for event in injection])
    assert all([event.final_states[0].pdg_code == 13 for event in injection])
    assert all([event.final_states[1].pdg_code == -2000001006 for event in injection])
    assert sum([event.bjorken_x for event in injection]) - 5.38459932518872 < 1e-5
    assert sum([event.bjorken_y for event in injection]) - 15.304668192727698 < 1e-5
    assert sum([event.column_depth for event in injection]) - 5398104.596123365 < 1e-5
    assert sum([event.vertex_x for event in injection]) + 903.6668603488667 < 1e-5
    assert sum([event.vertex_y for event in injection]) - 1720.581858684919 < 1e-5
    assert sum([event.vertex_z for event in injection]) + 97267.71623169626 < 1e-5
    assert sum([event.initial_state.e for event in injection]) - 5805025.309993334 < 1e-5
    assert sum(([event.initial_state.phi for event in injection])) - 24.598147958495357 < 1e-5
    assert sum(([event.initial_state.theta for event in injection])) - 72.11030193309311 < 1e-5
    assert np.sum([event.initial_state.position for event in injection]) + 96450.80123336022 < 1e-5
    assert np.sum([event.initial_state.direction for event in injection]) - 9.578980695601722 < 1e-5
    assert (
        np.sum([event.final_states[0].direction for event in injection]) - 1.4888529313548755 < 1e-5
    )
    assert (
        np.sum([event.final_states[1].direction for event in injection]) - 0.6596348700869346 < 1e-5
    )
    assert (
        np.sum([event.final_states[0].position for event in injection])
        - np.sum([event.initial_state.position for event in injection])
        < 1e-5
    )
    assert (
        np.sum([event.final_states[1].position for event in injection])
        - np.sum([event.initial_state.position for event in injection])
        < 1e-5
    )


def test_random_rotation_uniform_orthonormal() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        rot = _random_rotation(rng)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_genie_direction_mode_default_matches_as_is() -> None:
    cfg_missing = _genie_sim_config()
    del cfg_missing.direction_mode
    as_is = injection_from_genie_output(GENIE_FILE, simulation_config=_genie_sim_config())
    default = injection_from_genie_output(GENIE_FILE, simulation_config=cfg_missing)
    for ev_a, ev_d in zip(as_is, default):
        for fs_a, fs_d in zip(ev_a.final_states, ev_d.final_states):
            assert np.allclose(fs_a.direction, fs_d.direction)
            assert fs_a.e == fs_d.e


def test_genie_isotropic_preserves_kinematics() -> None:
    as_is = injection_from_genie_output(GENIE_FILE, simulation_config=_genie_sim_config())
    iso = injection_from_genie_output(
        GENIE_FILE, simulation_config=_genie_sim_config(direction_mode="isotropic")
    )
    assert len(as_is) == len(iso)
    for ev_a, ev_i in zip(as_is, iso):
        assert ev_a.initial_state.e == ev_i.initial_state.e
        pdgs_a = [fs.pdg_code for fs in ev_a.final_states]
        pdgs_i = [fs.pdg_code for fs in ev_i.final_states]
        assert pdgs_a == pdgs_i

        # A common rotation preserves every angle between final states.
        # Skip photons: pi0 decay draws its rest-frame angles from the RNG,
        # whose stream differs between the two modes.
        dirs_a = [fs.direction for fs in ev_a.final_states if fs.pdg_code != 22]
        dirs_i = [fs.direction for fs in ev_i.final_states if fs.pdg_code != 22]
        for j in range(len(dirs_a)):
            for k in range(j + 1, len(dirs_a)):
                assert np.isclose(
                    np.dot(dirs_a[j], dirs_a[k]), np.dot(dirs_i[j], dirs_i[k]), atol=1e-9
                )


def test_genie_isotropic_directions_are_isotropic() -> None:
    # Resample the 10-event file to 300 events; each copy receives an
    # independent rotation, so any fixed final state's cos(zenith) should be
    # uniform in [-1, 1] across events.
    iso = injection_from_genie_output(
        GENIE_FILE,
        simulation_config=_genie_sim_config(direction_mode="isotropic", n_events=300),
    )
    cos_z = np.array([np.cos(ev.final_states[0].theta) for ev in iso])
    assert abs(cos_z.mean()) < 0.15
    assert cos_z.min() < -0.5 and cos_z.max() > 0.5
    assert 0.3 < np.mean(cos_z > 0) < 0.7


def test_genie_unknown_direction_mode_raises() -> None:
    with pytest.raises(ValueError, match="direction_mode"):
        injection_from_genie_output(
            GENIE_FILE, simulation_config=_genie_sim_config(direction_mode="sideways")
        )
