"""Unit tests for the mDOM response model (dom_response / pmt_response /
fadc_digitization).
"""

import numpy as np

from prometheus.utils.dom_response import (
    PMT_DIRS,
    assign_to_pmts_per_hit,
    generate_fadc_response,
    process_event,
)

# ---------------------------------------------------------------------------
# generate_fadc_response
# ---------------------------------------------------------------------------


class TestGenerateFadcResponse:
    def test_qe_zero_detects_nothing(self):
        rng = np.random.default_rng(1)
        photon_times = np.sort(rng.uniform(0, 500, 100))
        fadc_t, fadc_q, n_pe, hit_t, tot_ns = generate_fadc_response(
            photon_times, qe=0.0, dark_rate_hz=0.0, rng=rng
        )
        assert n_pe == 0
        assert len(fadc_t) == 0
        assert len(hit_t) == 0

    def test_qe_one_detects_all_before_noise(self):
        rng = np.random.default_rng(2)
        photon_times = np.sort(rng.uniform(0, 500, 100))
        _, _, n_pe, _, _ = generate_fadc_response(photon_times, qe=1.0, dark_rate_hz=0.0, rng=rng)
        assert n_pe == 100

    def test_dark_noise_adds_hits_with_no_signal(self):
        rng = np.random.default_rng(3)
        fadc_t, fadc_q, n_pe, hit_t, tot_ns = generate_fadc_response(
            np.array([]), qe=1.0, dark_rate_hz=1e7, rng=rng
        )
        assert n_pe == 0
        assert len(fadc_t) > 0

    def test_fadc_charge_bins_are_non_negative(self):
        rng = np.random.default_rng(4)
        photon_times = np.sort(rng.uniform(0, 500, 200))
        fadc_t, fadc_q, n_pe, hit_t, tot_ns = generate_fadc_response(
            photon_times, qe=0.5, dark_rate_hz=750.0, rng=rng
        )
        assert np.all(fadc_q >= 0.0)
        assert np.all(tot_ns >= 1.0) and np.all(tot_ns <= 255.0)


# ---------------------------------------------------------------------------
# assign_to_pmts_per_hit
# ---------------------------------------------------------------------------


class TestAssignToPmtsPerHit:
    def test_photon_count_is_conserved(self):
        rng = np.random.default_rng(5)
        n = 300
        photon_times = rng.uniform(0, 500, n)
        source_dirs = np.tile([0.0, 0.0, 1.0], (n, 1))
        pmt_hits = assign_to_pmts_per_hit(photon_times, source_dirs, PMT_DIRS, qe=1.0, rng=rng)
        assert sum(len(v) for v in pmt_hits.values()) == n

    def test_qe_zero_assigns_nothing(self):
        rng = np.random.default_rng(6)
        n = 100
        photon_times = rng.uniform(0, 500, n)
        source_dirs = np.tile([0.0, 0.0, 1.0], (n, 1))
        pmt_hits = assign_to_pmts_per_hit(photon_times, source_dirs, PMT_DIRS, qe=0.0, rng=rng)
        assert pmt_hits == {}

    def test_direction_changes_illuminated_pmts(self):
        # A photon arriving from the +z direction should preferentially
        # light PMTs whose normal points toward +z, and vice versa for -z.
        rng = np.random.default_rng(7)
        n = 500
        times = np.zeros(n)
        up = np.tile([0.0, 0.0, 1.0], (n, 1))
        down = np.tile([0.0, 0.0, -1.0], (n, 1))

        hits_up = assign_to_pmts_per_hit(times, up, PMT_DIRS, qe=1.0, rng=rng)
        hits_down = assign_to_pmts_per_hit(times, down, PMT_DIRS, qe=1.0, rng=rng)

        def mean_pz(hits):
            idx = np.array(list(hits.keys()))
            counts = np.array([len(v) for v in hits.values()])
            return np.average(PMT_DIRS[idx, 2], weights=counts)

        assert mean_pz(hits_up) > mean_pz(hits_down)


# ---------------------------------------------------------------------------
# process_event
# ---------------------------------------------------------------------------


class TestProcessEvent:
    def _make_photons(self, rng, n=200):
        return {
            "string_id": rng.integers(0, 3, n),
            "sensor_id": rng.integers(0, 5, n),
            "t": rng.uniform(0, 500, n),
            "sensor_pos_x": rng.integers(0, 3, n).astype(float) * 10.0,
            "sensor_pos_y": rng.integers(0, 5, n).astype(float) * 10.0,
            "sensor_pos_z": np.zeros(n),
        }

    def test_output_has_expected_keys(self):
        rng = np.random.default_rng(8)
        photons = self._make_photons(rng)
        result = process_event(photons, np.array([5.0, 5.0, -20.0]), rng=rng)
        assert set(result) == {
            "string_id",
            "sensor_id",
            "sensor_pos_x",
            "sensor_pos_y",
            "sensor_pos_z",
            "pmt_id",
            "pmt_dir_x",
            "pmt_dir_y",
            "pmt_dir_z",
            "n_pe",
            "fadc_t",
            "fadc_q",
            "hit_t",
            "tot_ns",
        }

    def test_empty_event_produces_no_pmt_entries(self):
        rng = np.random.default_rng(9)
        photons = {
            "string_id": np.array([]),
            "sensor_id": np.array([]),
            "t": np.array([]),
            "sensor_pos_x": np.array([]),
            "sensor_pos_y": np.array([]),
            "sensor_pos_z": np.array([]),
        }
        result = process_event(photons, np.array([0.0, 0.0, 0.0]), rng=rng)
        assert len(result["pmt_id"]) == 0

    def test_every_fired_pmt_id_is_valid(self):
        rng = np.random.default_rng(10)
        photons = self._make_photons(rng)
        result = process_event(photons, np.array([5.0, 5.0, -20.0]), rng=rng)
        assert all(0 <= pid < len(PMT_DIRS) for pid in result["pmt_id"])
