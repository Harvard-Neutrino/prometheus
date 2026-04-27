"""Unit tests for prometheus.utils.geo_utils — geofile I/O and round-trips."""

import pathlib

import numpy as np
import pytest

from prometheus.utils.geo_utils import from_geo, geo_from_coords

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
WATER_GEO = str(REPO_ROOT / "resources/geofiles/demo_water.geo")
ICE_GEO = str(REPO_ROOT / "resources/geofiles/demo_ice.geo")
ICECUBE_GEO = str(REPO_ROOT / "resources/geofiles/icecube.geo")
PONE_GEO = str(REPO_ROOT / "resources/geofiles/pone_triangle.geo")


# ---------------------------------------------------------------------------
# from_geo — read an existing geometry file
# ---------------------------------------------------------------------------


class TestFromGeoWater:
    @pytest.fixture(scope="class")
    def parsed(self):
        return from_geo(WATER_GEO)

    def test_returns_three_values(self, parsed):
        assert len(parsed) == 3

    def test_positions_are_ndarray(self, parsed):
        pos, _keys, _medium = parsed
        assert isinstance(pos, np.ndarray)

    def test_positions_shape_nx3(self, parsed):
        pos, _keys, _medium = parsed
        assert pos.ndim == 2
        assert pos.shape[1] == 3

    def test_keys_are_list_of_tuples(self, parsed):
        _pos, keys, _medium = parsed
        assert isinstance(keys, list)
        for k in keys:
            assert isinstance(k, tuple)
            assert len(k) == 2

    def test_key_elements_are_ints(self, parsed):
        _pos, keys, _medium = parsed
        for string_id, om_id in keys:
            assert isinstance(string_id, int)
            assert isinstance(om_id, int)

    def test_positions_and_keys_same_length(self, parsed):
        pos, keys, _medium = parsed
        assert len(pos) == len(keys)

    def test_medium_is_water(self, parsed):
        _pos, _keys, medium = parsed
        assert medium.lower() == "water"

    def test_positive_module_count(self, parsed):
        pos, _keys, _medium = parsed
        assert len(pos) > 0


class TestFromGeoIce:
    @pytest.fixture(scope="class")
    def parsed(self):
        return from_geo(ICE_GEO)

    def test_medium_is_ice(self, parsed):
        _pos, _keys, medium = parsed
        assert medium.lower() == "ice"

    def test_has_modules(self, parsed):
        pos, _keys, _medium = parsed
        assert len(pos) > 0

    def test_positions_shape_nx3(self, parsed):
        pos, _keys, _medium = parsed
        assert pos.shape[1] == 3


class TestFromGeoIceCube:
    """IceCube geometry has many strings; basic sanity checks."""

    @pytest.fixture(scope="class")
    def parsed(self):
        return from_geo(ICECUBE_GEO)

    def test_medium_is_ice(self, parsed):
        _pos, _keys, medium = parsed
        assert medium.lower() == "ice"

    def test_module_count_plausible(self, parsed):
        # IceCube has 5160 OMs (86 strings × 60 DOMs)
        pos, _keys, _medium = parsed
        assert len(pos) > 1000

    def test_positions_span_reasonable_z_range(self, parsed):
        pos, _keys, _medium = parsed
        z_span = pos[:, 2].max() - pos[:, 2].min()
        # IceCube spans ~1000 m vertically
        assert z_span > 500


class TestFromGeoPOne:
    @pytest.fixture(scope="class")
    def parsed(self):
        return from_geo(PONE_GEO)

    def test_medium_is_water(self, parsed):
        _pos, _keys, medium = parsed
        assert medium.lower() == "water"

    def test_has_modules(self, parsed):
        pos, _keys, _medium = parsed
        assert len(pos) > 0


# ---------------------------------------------------------------------------
# geo_from_coords — write a geometry and read it back
# ---------------------------------------------------------------------------


class TestGeoFromCoordsRoundTrip:
    """geo_from_coords writes valid geofiles that from_geo can re-read."""

    @pytest.fixture(scope="class")
    def simple_coords(self):
        """3×3 grid of 5 modules each, at z=0, 1, 2, 3, 4."""
        coords = []
        for x in [0.0, 10.0, 20.0]:
            for y in [0.0, 10.0, 20.0]:
                for z in range(5):
                    coords.append([float(x), float(y), float(z)])
        return np.array(coords)

    def test_roundtrip_module_count(self, simple_coords, tmp_path):
        out = str(tmp_path / "test.geo")
        geo_from_coords(simple_coords, out, medium="water")
        pos, keys, medium = from_geo(out)
        assert len(pos) == len(simple_coords)

    def test_roundtrip_medium_preserved(self, simple_coords, tmp_path):
        out = str(tmp_path / "test.geo")
        geo_from_coords(simple_coords, out, medium="ice")
        _pos, _keys, medium = from_geo(out)
        assert medium.lower() == "ice"

    def test_roundtrip_positions_close(self, simple_coords, tmp_path):
        out = str(tmp_path / "test.geo")
        geo_from_coords(simple_coords, out, medium="water")
        pos, _keys, _medium = from_geo(out)
        # Positions may be reordered by geo_from_coords (it sorts); compare sets
        orig_set = {tuple(np.round(c, 3)) for c in simple_coords}
        read_set = {tuple(np.round(p, 3)) for p in pos}
        assert orig_set == read_set

    def test_roundtrip_keys_are_valid_tuples(self, simple_coords, tmp_path):
        out = str(tmp_path / "test.geo")
        geo_from_coords(simple_coords, out, medium="water")
        _pos, keys, _medium = from_geo(out)
        for k in keys:
            assert isinstance(k, tuple)
            assert len(k) == 2

    def test_single_module_roundtrip(self, tmp_path):
        coords = np.array([[0.0, 0.0, 0.0]])
        out = str(tmp_path / "single.geo")
        geo_from_coords(coords, out, medium="water")
        pos, keys, medium = from_geo(out)
        assert len(pos) == 1
        assert medium.lower() == "water"

    def test_dom_radius_written_to_file(self, tmp_path):
        coords = np.array([[0.0, 0.0, 0.0]])
        out = str(tmp_path / "radius.geo")
        geo_from_coords(coords, out, medium="water", dom_radius=42)
        with open(out) as f:
            content = f.read()
        assert "42" in content
