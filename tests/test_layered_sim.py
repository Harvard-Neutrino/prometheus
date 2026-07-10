"""Unit tests for the layered-detector simulation helpers
(layered_sim / detector_layers).

Imports go through ``prometheus.utils.layered_sim`` -- the module that
re-exports the geometry helpers from ``detector_layers`` -- so these tests
also guard the backward-compatible import surface relied on by the example
scripts.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from prometheus.detector.detector_factory import make_grid
from prometheus.detector.medium import Medium
from prometheus.utils.layered_sim import (
    build_layers,
    cell_geometry,
    cell_manifest_dict,
    count_hits_particle,
    event_stats,
    extract_strings,
    partition_layers,
    representative_string,
    sample_cell_vertices,
    wilson_sigma,
)


@pytest.fixture
def grid_detector():
    # 4x4 square grid of strings, 100 m spacing, 3 modules/string spanning
    # z in [-10, 10].
    return make_grid(
        n_side=4, dist=100.0, n_z=3, dist_z=10.0, z_cent=0.0, medium=Medium.WATER
    )


# ---------------------------------------------------------------------------
# extract_strings / cell_geometry
# ---------------------------------------------------------------------------


class TestExtractStrings:
    def test_returns_one_row_per_string(self, grid_detector):
        strings = extract_strings(grid_detector)
        assert strings.shape == (16, 2)

    def test_string_xy_matches_module_xy(self, grid_detector):
        strings = extract_strings(grid_detector)
        string_ids = np.asarray([m.key[0] for m in grid_detector.modules])
        xy = grid_detector.module_coords[:, :2]
        for sid in np.unique(string_ids):
            expected = xy[string_ids == sid][0]
            assert np.any(np.all(np.isclose(strings, expected), axis=1))


class TestCellGeometry:
    def test_n_strings_matches_grid(self, grid_detector):
        strings = extract_strings(grid_detector)
        cell = cell_geometry(grid_detector, strings)
        assert cell.n_strings == 16

    def test_z_extent_matches_module_span(self, grid_detector):
        strings = extract_strings(grid_detector)
        cell = cell_geometry(grid_detector, strings)
        coords = grid_detector.module_coords
        assert cell.z_min == pytest.approx(coords[:, 2].min())
        assert cell.z_max == pytest.approx(coords[:, 2].max())
        assert cell.H == pytest.approx(cell.z_max - cell.z_min)

    def test_r_det_extends_past_outermost_string(self, grid_detector):
        strings = extract_strings(grid_detector)
        cell = cell_geometry(grid_detector, strings)
        assert cell.R_det > cell.R_outer

    def test_v_cell_totals_to_full_footprint(self, grid_detector):
        strings = extract_strings(grid_detector)
        cell = cell_geometry(grid_detector, strings)
        total = cell.V_cell * cell.n_strings
        assert total == pytest.approx(np.pi * cell.R_det**2 * cell.H)

    def test_r_cell_relates_to_r_det(self, grid_detector):
        strings = extract_strings(grid_detector)
        cell = cell_geometry(grid_detector, strings)
        assert cell.r_cell == pytest.approx(np.sqrt(cell.R_det**2 / cell.n_strings))


# ---------------------------------------------------------------------------
# partition_layers / representative_string / build_layers
# ---------------------------------------------------------------------------


class TestPartitionLayers:
    def test_layers_partition_all_indices_exactly_once(self):
        radii = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 0.0])
        layers = partition_layers(radii, n_layers=3)
        all_indices = np.concatenate(layers)
        assert sorted(all_indices.tolist()) == list(range(len(radii)))

    def test_layers_ordered_by_increasing_radius(self):
        radii = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 0.0])
        layers = partition_layers(radii, n_layers=3)
        assert radii[layers[0]].max() <= radii[layers[-1]].min()

    def test_n_layers_respected(self):
        radii = np.arange(10.0)
        layers = partition_layers(radii, n_layers=4)
        assert len(layers) == 4


class TestRepresentativeString:
    def test_picks_closest_to_median_radius(self):
        strings = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        radii = np.array([0.0, 1.0, 2.0, 3.0])
        indices = np.array([0, 1, 2, 3])
        rep = representative_string(strings, radii, indices)
        # Median radius is 1.5; ties between radius 1.0 and 2.0 broken by
        # argmin -> first match, radius 1.0 -> strings[1].
        assert np.allclose(rep, strings[1])

    def test_restricted_to_given_indices(self):
        strings = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        radii = np.array([0.0, 1.0, 2.0, 3.0])
        indices = np.array([2, 3])
        rep = representative_string(strings, radii, indices)
        assert np.allclose(rep, strings[2])


class TestBuildLayers:
    def test_layer_string_counts_sum_to_total(self, grid_detector):
        layers, cell = build_layers(grid_detector, n_layers=4)
        assert sum(layer["n_strings"] for layer in layers) == cell.n_strings

    def test_each_layer_has_expected_keys(self, grid_detector):
        layers, _ = build_layers(grid_detector, n_layers=4)
        for layer in layers:
            assert set(layer) == {"n_strings", "r_range", "rep_xy"}
            assert layer["rep_xy"].shape == (2,)

    def test_r_ranges_increase_across_layers(self, grid_detector):
        layers, _ = build_layers(grid_detector, n_layers=4)
        maxima = [layer["r_range"][1] for layer in layers]
        assert maxima == sorted(maxima)


class TestCellManifestDict:
    def test_keys_and_values_match_cell(self, grid_detector):
        strings = extract_strings(grid_detector)
        cell = cell_geometry(grid_detector, strings)
        manifest = cell_manifest_dict(cell)
        assert manifest == {
            "n_strings": cell.n_strings,
            "R_outer_m": cell.R_outer,
            "R_det_m": cell.R_det,
            "d_nn_m": cell.d_nn,
            "V_cell_m3": cell.V_cell,
            "r_cell_m": cell.r_cell,
            "z_min_m": cell.z_min,
            "z_max_m": cell.z_max,
            "H_m": cell.H,
        }


# ---------------------------------------------------------------------------
# sample_cell_vertices
# ---------------------------------------------------------------------------


class TestSampleCellVertices:
    def test_points_stay_within_cell(self):
        rng = np.random.default_rng(11)
        string_xy = np.array([50.0, -20.0])
        r_cell, z_min, z_max = 30.0, -100.0, 100.0
        pts = sample_cell_vertices(rng, string_xy, r_cell, z_min, z_max, 500)
        assert pts.shape == (500, 3)
        radial = np.linalg.norm(pts[:, :2] - string_xy, axis=1)
        assert np.all(radial <= r_cell + 1e-9)
        assert np.all(pts[:, 2] >= z_min) and np.all(pts[:, 2] <= z_max)

    def test_zero_count_returns_empty(self):
        rng = np.random.default_rng(12)
        pts = sample_cell_vertices(rng, np.array([0.0, 0.0]), 10.0, -5.0, 5.0, 0)
        assert pts.shape == (0, 3)


# ---------------------------------------------------------------------------
# wilson_sigma
# ---------------------------------------------------------------------------


class TestWilsonSigma:
    def test_zero_trials_returns_zero(self):
        assert wilson_sigma(0, 0) == 0.0

    def test_known_value_all_successes(self):
        # k == n collapses k*(n-k)/n to 0, leaving z*sqrt(z^2/4)/(n+z^2).
        assert wilson_sigma(100, 100, z=1.0) == pytest.approx(0.0049504950495049506)

    def test_known_value_all_failures_matches_all_successes(self):
        # Symmetric under k -> n - k.
        assert wilson_sigma(0, 100, z=1.0) == pytest.approx(0.0049504950495049506)

    def test_known_value_half_successes(self):
        assert wilson_sigma(50, 100, z=1.0) == pytest.approx(0.04975185951049946)

    def test_known_value_nondefault_z(self):
        assert wilson_sigma(3, 20, z=1.959963984540054) == pytest.approx(
            0.15402505942227016
        )


# ---------------------------------------------------------------------------
# count_hits_particle / event_stats
# ---------------------------------------------------------------------------


def _hit(string_id, om_id):
    return SimpleNamespace(string_id=string_id, om_id=om_id)


class TestCountHitsParticle:
    def test_counts_own_hits_only(self):
        particle = SimpleNamespace(hits=[_hit(0, 1), _hit(0, 2)], children=[])
        n_hits, modules = count_hits_particle(particle)
        assert n_hits == 2
        assert modules == {(0, 1), (0, 2)}

    def test_includes_children_recursively(self):
        grandchild = SimpleNamespace(hits=[_hit(2, 0)], children=[])
        child = SimpleNamespace(hits=[_hit(1, 0)], children=[grandchild])
        particle = SimpleNamespace(hits=[_hit(0, 0)], children=[child])
        n_hits, modules = count_hits_particle(particle)
        assert n_hits == 3
        assert modules == {(0, 0), (1, 0), (2, 0)}

    def test_duplicate_module_hits_count_once_in_module_set(self):
        # Two hits on the same module contribute 2 to n_hits but 1 module.
        particle = SimpleNamespace(hits=[_hit(0, 0), _hit(0, 0)], children=[])
        n_hits, modules = count_hits_particle(particle)
        assert n_hits == 2
        assert modules == {(0, 0)}

    def test_missing_hits_and_children_attrs_treated_as_empty(self):
        particle = SimpleNamespace()
        n_hits, modules = count_hits_particle(particle)
        assert n_hits == 0
        assert modules == set()


class TestEventStats:
    def test_sums_hits_and_modules_across_final_states(self):
        fs1 = SimpleNamespace(hits=[_hit(0, 0), _hit(0, 1)], children=[])
        fs2 = SimpleNamespace(hits=[_hit(0, 1), _hit(1, 0)], children=[])
        event = SimpleNamespace(final_states=[fs1, fs2])
        hit_counts, module_counts = event_stats([event])
        assert hit_counts.tolist() == [4]
        # (0,0), (0,1), (1,0) -- (0,1) shared between the two final states.
        assert module_counts.tolist() == [3]

    def test_multiple_events(self):
        fs_a = SimpleNamespace(hits=[_hit(0, 0)], children=[])
        fs_b = SimpleNamespace(hits=[], children=[])
        event_a = SimpleNamespace(final_states=[fs_a])
        event_b = SimpleNamespace(final_states=[fs_b])
        hit_counts, module_counts = event_stats([event_a, event_b])
        assert hit_counts.tolist() == [1, 0]
        assert module_counts.tolist() == [1, 0]
