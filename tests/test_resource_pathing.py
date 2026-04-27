"""Regression tests for resource directory discovery.

Guards against the bug where RESOURCES_DIR resolved to site-packages/resources/
instead of the repository root when Prometheus was installed via modern pip
editable installs (PEP 660).
"""

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# RESOURCES_DIR resolution
# ---------------------------------------------------------------------------


class TestResourcesDirResolution:
    """RESOURCES_DIR resolves to the correct repository location."""

    def test_config_mims_resources_dir_is_directory(self):
        from prometheus.utils.config_mims import RESOURCES_DIR

        assert RESOURCES_DIR.is_dir(), f"RESOURCES_DIR does not exist: {RESOURCES_DIR}"

    def test_config_types_resources_dir_is_directory(self):
        from prometheus.config_types import RESOURCES_DIR

        assert RESOURCES_DIR.is_dir(), f"RESOURCES_DIR does not exist: {RESOURCES_DIR}"

    def test_both_modules_agree_on_resources_dir(self):
        from prometheus.config_types import RESOURCES_DIR as rt
        from prometheus.utils.config_mims import RESOURCES_DIR as rm

        assert rm == rt

    def test_cross_section_splines_exist(self):
        from prometheus.utils.config_mims import RESOURCES_DIR

        xsec_dir = RESOURCES_DIR / "cross_section_splines"
        assert xsec_dir.is_dir(), f"cross_section_splines/ not found under {RESOURCES_DIR}"

    def test_required_xsec_files_exist(self):
        from prometheus.utils.config_mims import RESOURCES_DIR

        xsec_dir = RESOURCES_DIR / "cross_section_splines"
        for fname in (
            "dsdxdy_nu_CC_iso.fits",
            "dsdxdy_nubar_CC_iso.fits",
            "sigma_nu_CC_iso.fits",
            "sigma_nubar_CC_iso.fits",
        ):
            assert (xsec_dir / fname).is_file(), f"Missing cross-section file: {fname}"

    def test_earthparams_exist(self):
        from prometheus.utils.config_mims import RESOURCES_DIR

        assert (RESOURCES_DIR / "earthparams").is_dir()

    def test_resources_dir_not_in_site_packages(self):
        from prometheus.utils.config_mims import RESOURCES_DIR

        assert "site-packages" not in str(RESOURCES_DIR), (
            f"RESOURCES_DIR resolved into site-packages: {RESOURCES_DIR}"
        )


# ---------------------------------------------------------------------------
# _find_resources_dir walk-up logic
# ---------------------------------------------------------------------------


class TestFindResourcesDirLogic:
    """The walk-up algorithm finds resources/ and fails gracefully when absent."""

    def _run_algorithm(self, start: Path) -> Path:
        """Re-implement the walk-up algorithm starting from an arbitrary path."""
        for parent in start.resolve().parents:
            candidate = parent / "resources"
            if candidate.is_dir():
                return candidate
        raise RuntimeError("Cannot find the Prometheus resources/ directory.")

    def test_finds_resources_when_nested_inside_repo(self, tmp_path):
        (tmp_path / "resources").mkdir()
        deep = tmp_path / "lib" / "python3.12" / "site-packages" / "pkg" / "utils"
        deep.mkdir(parents=True)

        result = self._run_algorithm(deep / "some_module.py")

        assert result == tmp_path / "resources"

    def test_raises_when_no_resources_dir_exists(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="Cannot find the Prometheus resources"):
            self._run_algorithm(deep / "module.py")
