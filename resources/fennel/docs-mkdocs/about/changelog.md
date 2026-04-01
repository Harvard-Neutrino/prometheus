# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PR guidance: new documentation at `docs-mkdocs/development/pr-guide.md` outlining branching, testing, changelog, and review expectations
- GitHub Actions workflow for automatic documentation deployment to GitHub Pages
- MkDocs Material documentation is now the official documentation system

### Changed
- **Documentation Migration**: Transitioned from Sphinx (in `docs/`) to MkDocs Material (in `docs-mkdocs/`)
  - New documentation site: https://meighenbergers.github.io/fennel/
  - Auto-deployed via GitHub Actions on every push to master
  - Old Sphinx documentation in `docs/` is now deprecated (see `docs/DEPRECATED.md`)
- Updated README.md with comprehensive documentation links and improved structure
- Added documentation build commands to Makefile (`make docs-serve`, `make docs-build`)

### Deprecated
- Sphinx documentation in `docs/` directory - use MkDocs documentation instead

## [2.0.0] - 2026-01-09

### Added
- **Result Container Classes**: New `TrackYieldResult`, `EMYieldResult`, and `HadronYieldResult` classes that provide structured, self-documenting results with named attributes
- **v2 API Methods**: 
  - `track_yields_v2()`: Returns `TrackYieldResult` instead of tuple
  - `em_yields_v2()`: Returns `EMYieldResult` with additional metadata
  - `hadron_yields_v2()`: Returns `HadronYieldResult` with EM fraction info
- **Convenience Methods**:
  - `quick_track()`: Simplified muon track calculations
  - `quick_cascade()`: Simplified cascade calculations
  - `calculate()`: Universal method that auto-detects particle type
- **Input Validation**: Comprehensive validation at API boundaries with helpful error messages via `ValidationError` exception
- **Validation Module**: New `fennel.validation` module with validators for energy, particle PDG codes, wavelengths, angles, and refractive indices
- **Pretty Printing**: Result containers have informative `__repr__` methods showing all contained data
- **Particle Names**: Result objects include human-readable particle names (e.g., "e⁻", "π⁺")
- **Comprehensive Test Suite**: 123 tests covering v2 API, validation, result containers, convenience methods, and backward compatibility
- **Documentation**: 
  - Complete `UPGRADE_GUIDE_V2.md` with migration examples
  - Example notebook `example_v2.ipynb` demonstrating all new features
  - Extended docstrings for all new methods
- **GitHub Actions**: CI workflow for automated testing on push and PR
- **Pre-commit Hooks**: Configuration for code quality tools
- **MkDocs Documentation**: Modern documentation site with API reference and user guide

### Changed
- **Major Version Bump**: Version 2.0.0 to indicate new API surface (fully backward compatible)
- All internal modules now use consistent validation via `fennel.validation`
- Improved error messages across the codebase
- Enhanced type hints throughout

### Deprecated
- None (old API remains fully supported)

### Fixed
- More robust input validation prevents silent errors
- Consistent handling of edge cases across all methods

### Backward Compatibility
- **100% Compatible**: All v1.x methods work identically
- No breaking changes to existing API
- Old tuple-unpacking code continues to work
- Users can migrate gradually or not at all

---

## [1.3.4] - Previous Release

(Prior changelog entries would go here)
