"""Unit tests for the memory behaviour of the olympus photon propagation path.

These cover the two rewrites that bound memory on a densely instrumented
detector -- chunked evaluation of the model-input kernel, and the chunked
source/module range pre-filter -- plus the smaller retained-size and
cache-release pieces. The rewrites exist for memory reasons only, so the tests
that matter most assert that they are numerically indistinguishable from the
whole-array computations they replaced.
"""

import numpy as np
import pytest

import prometheus.photon_propagation.olympus.event_generation.event_generation as event_generation
from prometheus.config_types import OlympusSimConfig, RunConfig
from prometheus.photon_propagation.hit import Hit
from prometheus.photon_propagation.olympus.event_generation.event_generation import (
    _in_range_masks,
)
from prometheus.photon_propagation.olympus.event_generation.photon_propagation.utils import (
    next_bucket,
    sources_to_model_input,
    sources_to_model_input_chunked,
)
from prometheus.utils.memory import release_propagator_memory


def _random_sources(rng, n_src):
    """Build random source positions, directions and emission times."""
    source_dir = rng.normal(0, 1, (n_src, 3))
    norms = np.linalg.norm(source_dir, axis=1, keepdims=True)
    source_dir = np.divide(source_dir, norms, out=np.zeros_like(source_dir), where=norms > 0)
    return (
        rng.normal(0, 50, (n_src, 3)),
        source_dir,
        rng.normal(0, 10, (n_src, 1)),
    )


# ---------------------------------------------------------------------------
# sources_to_model_input_chunked
# ---------------------------------------------------------------------------


class TestSourcesToModelInputChunked:
    @pytest.mark.parametrize(
        "n_mod, n_src, chunk",
        [
            (1, 1, 128),  # single module, single source
            (7, 3, 128),  # fewer modules than one chunk
            (128, 4, 128),  # exactly one full chunk
            (300, 8, 128),  # ragged final chunk
            (513, 6, 128),  # several chunks, ragged tail
            (64, 5, 1),  # degenerate chunk size
        ],
    )
    def test_matches_whole_array(self, n_mod, n_src, chunk):
        """Chunking the module axis must not change the computed values."""
        rng = np.random.default_rng(0)
        module_coords = rng.normal(0, 200, (n_mod, 3))
        source_pos, source_dir, source_time = _random_sources(rng, n_src)

        want_inp, want_time = sources_to_model_input(
            module_coords, source_pos, source_dir, source_time, 0.2
        )
        got_inp, got_time = sources_to_model_input_chunked(
            module_coords, source_pos, source_dir, source_time, 0.2, chunk
        )
        want_inp = np.asarray(want_inp)
        want_time = np.asarray(want_time)

        assert got_inp.shape == want_inp.shape
        assert got_time.shape == want_time.shape

        # Each (module, source) pair is evaluated independently of every
        # other, so chunking that axis is exact in real arithmetic. It is not
        # exact in floating point: XLA picks its vectorisation from the array
        # shape, so a (n_src, 128, 3) block and a (n_src, 300, 3) one can
        # contract differently and disagree in the last bit or two. Which way
        # that falls depends on the CPU and the jaxlib build, so asserting
        # bitwise equality here passes on one machine and fails on another.
        #
        # The tolerance is scaled off the working dtype and is still many
        # orders of magnitude tighter than anything physically meaningful.
        # The failures this guards against -- a scrambled module order, a
        # dropped or duplicated chunk, a mistrimmed pad -- all move values by
        # order unity, so they are caught regardless.
        rtol = 1e3 * np.finfo(got_inp.dtype).eps
        np.testing.assert_allclose(got_inp, want_inp, rtol=rtol, atol=rtol)
        np.testing.assert_allclose(got_time, want_time, rtol=rtol, atol=rtol)

    def test_no_modules_returns_empty(self):
        rng = np.random.default_rng(0)
        source_pos, source_dir, source_time = _random_sources(rng, 4)
        inp, time_geo = sources_to_model_input_chunked(
            np.zeros((0, 3)), source_pos, source_dir, source_time, 0.2, 128
        )
        assert inp.shape == (4, 0, 2)
        assert time_geo.shape == (4, 0, 1)

    def test_compiled_variants_do_not_grow_with_module_count(self):
        """The point of the rewrite: cache size independent of detector size.

        Bucketing both axes makes the number of cached executables the
        product of the two ladders. Holding the module axis fixed leaves only
        the source bucket to vary, so a bigger detector costs no extra
        compiled variants.
        """
        rng = np.random.default_rng(2)
        src_counts = [16, 64, 256]

        sources_to_model_input._clear_cache()
        try:
            seen = []
            for n_mod in (256, 2048, 8192):
                for n_src in src_counts:
                    source_pos, source_dir, source_time = _random_sources(
                        rng, next_bucket(n_src, minimum=16)
                    )
                    sources_to_model_input_chunked(
                        rng.normal(0, 200, (n_mod, 3)),
                        source_pos,
                        source_dir,
                        source_time,
                        0.2,
                        128,
                    )
                seen.append(sources_to_model_input._cache_size())

            assert seen[0] == len(src_counts)
            assert seen == [seen[0]] * len(seen)
        finally:
            sources_to_model_input._clear_cache()


# ---------------------------------------------------------------------------
# _in_range_masks
# ---------------------------------------------------------------------------


def _dense_reference(source_pos, module_coords, max_distance):
    """Whole-array range pre-filter, as it was before chunking."""
    dist_matrix = np.linalg.norm(
        np.asarray(source_pos)[:, np.newaxis, :] - module_coords[np.newaxis, :, :],
        axis=-1,
    )
    source_mask = np.any(dist_matrix < max_distance, axis=1)
    module_mask = np.any(dist_matrix[source_mask] < max_distance, axis=0)
    return source_mask, module_mask


class TestInRangeMasks:
    @pytest.mark.parametrize(
        "n_mod, n_src, max_distance",
        [
            (500, 40, 300.0),  # ordinary case, both masks partly true
            (2000, 200, 300.0),  # larger, forces several chunks
            (50, 5, 5.0),  # nothing in range
            (50, 5, 1e6),  # everything in range
        ],
    )
    def test_matches_dense_reference(self, n_mod, n_src, max_distance, monkeypatch):
        rng = np.random.default_rng(3)
        module_coords = rng.normal(0, 400, (n_mod, 3))
        source_pos = rng.normal(0, 400, (n_src, 3))

        want_src, want_mod = _dense_reference(source_pos, module_coords, max_distance)

        # Shrink the budget so the chunk loop runs several iterations even on
        # these small inputs.
        monkeypatch.setattr(
            event_generation, "_RANGE_FILTER_BUDGET_BYTES", max(1, n_mod * 3 * 8 * 3)
        )
        got_src, got_mod = _in_range_masks(source_pos, module_coords, max_distance)

        assert np.array_equal(got_src, want_src)
        assert np.array_equal(got_mod, want_mod)

    @pytest.mark.parametrize("n_mod, n_src", [(0, 10), (10, 0), (0, 0)])
    def test_empty_inputs(self, n_mod, n_src):
        rng = np.random.default_rng(4)
        source_mask, module_mask = _in_range_masks(
            rng.normal(0, 1, (n_src, 3)), rng.normal(0, 1, (n_mod, 3)), 300.0
        )
        assert source_mask.shape == (n_src,)
        assert module_mask.shape == (n_mod,)
        assert not source_mask.any()
        assert not module_mask.any()

    def test_chunking_does_not_depend_on_budget(self, monkeypatch):
        """Same answer whichever chunk size the budget happens to imply."""
        rng = np.random.default_rng(5)
        module_coords = rng.normal(0, 300, (400, 3))
        source_pos = rng.normal(0, 300, (60, 3))

        results = []
        for budget in (1, 400 * 24, 400 * 24 * 7, 1 << 30):
            monkeypatch.setattr(event_generation, "_RANGE_FILTER_BUDGET_BYTES", budget)
            results.append(_in_range_masks(source_pos, module_coords, 300.0))

        for source_mask, module_mask in results[1:]:
            assert np.array_equal(source_mask, results[0][0])
            assert np.array_equal(module_mask, results[0][1])


# ---------------------------------------------------------------------------
# Retained hit size and cache release
# ---------------------------------------------------------------------------


class TestHitLayout:
    def test_hit_is_slotted(self):
        """Hits stay resident for the whole run, so the layout is load-bearing."""
        hit = Hit(1, 2, 3.0, None, None, None, None, None)
        assert not hasattr(hit, "__dict__")
        with pytest.raises(AttributeError):
            hit.not_a_field = 1

    def test_hit_fields_round_trip(self):
        hit = Hit(1, 2, 3.0, 400.0, 0.1, 0.2, 0.3, 0.4)
        assert (hit.string_id, hit.om_id, hit.time) == (1, 2, 3.0)
        assert hit.wavelength == 400.0
        assert (hit.photon_zenith, hit.photon_azimuth) == (0.3, 0.4)


class TestReleasePropagatorMemory:
    def test_is_callable_and_preserves_results(self):
        """Releasing drops compiled code only; live arrays must survive."""
        rng = np.random.default_rng(6)
        module_coords = rng.normal(0, 200, (64, 3))
        source_pos, source_dir, source_time = _random_sources(rng, 8)

        before, _ = sources_to_model_input_chunked(
            module_coords, source_pos, source_dir, source_time, 0.2, 128
        )
        release_propagator_memory()
        after, _ = sources_to_model_input_chunked(
            module_coords, source_pos, source_dir, source_time, 0.2, 128
        )
        assert np.array_equal(before, after)

    def test_repeated_calls_are_safe(self):
        for _ in range(3):
            release_propagator_memory()


# ---------------------------------------------------------------------------
# Configuration surface
# ---------------------------------------------------------------------------


class TestConfigKnobs:
    def test_release_interval_defaults_off(self):
        assert RunConfig().jax_release_interval == 0

    def test_release_interval_accepts_spaced_key(self):
        run = RunConfig()
        run["jax release interval"] = 50
        assert run.jax_release_interval == 50

    def test_module_chunk_default_and_spaced_key(self):
        sim = OlympusSimConfig()
        assert sim.module_chunk == 128
        sim["module chunk"] = 256
        assert sim.module_chunk == 256

    def test_splitter_still_loads(self):
        """Retained purely so existing configuration files keep working."""
        sim = OlympusSimConfig()
        sim["splitter"] = 10000
        assert sim.splitter == 10000
