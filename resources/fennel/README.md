<img src="docs-mkdocs/assets/Fennel.png" alt="Fennel logo" width="180" />

# Fennel

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://meighenbergers.github.io/fennel/)

A Python package for simulating Cherenkov light yields from particle tracks and cascades in ice or water. The v2 API adds structured results, validation, and convenience helpers while keeping the classic v1 tuple API fully supported.

## Table of contents
- [What's New in v2.0.0](#whats-new-in-v200)
- [Features](#features)
- [Documentation](#documentation)
- [Installation](#installation)
- [Quickstart (v2 API)](#quickstart-v2-api)
- [Testing](#testing)
- [Development](#development)
- [Citation](#citation)
- [Support](#support)
- [Beta projects](#beta-projects)

## What's New in v2.0.0

🎉 **Major release with backward-compatible API improvements!**

- **Structured Results**: New result container classes (`TrackYieldResult`, `EMYieldResult`, `HadronYieldResult`) with named attributes and pretty printing
- **Convenience Methods**: One-line calculations with `quick_track()`, `quick_cascade()`, and `calculate()`
- **Input Validation**: Comprehensive validation with helpful error messages via `ValidationError`
- **Enhanced Documentation**: Complete MkDocs site with user guides, API reference, and examples
- **100% Backward Compatible**: All v1.x methods continue to work identically

See the full [Changelog](CHANGELOG.md) and [Upgrade Guide](https://meighenbergers.github.io/fennel/getting-started/upgrade-guide/) for details.

> **Note**: The v2.0.0 release is currently only available via GitHub installation. PyPI still hosts v1.3.4.

## Features

### Physics Coverage
- **Muon Tracks**: Energy loss mechanisms (ionization, bremsstrahlung, pair production) with wavelength-dependent yields
- **EM Cascades**: Electron, positron, and photon-initiated showers with longitudinal profiles and angular distributions
- **Hadron Cascades**: Pion, kaon, proton, and neutron showers with EM fraction tracking and muon production
- **Validated Physics**: Reference values from GEANT4 simulations ensure accuracy

### Technical Features
- **High Performance**: NumPy-based calculations optimized for CPU efficiency
- **GPU Acceleration**: Optional JAX backend for massive speedups on large batches
- **User-Friendly API**: Intuitive interface with comprehensive type hints and documentation
- **Flexible Configuration**: YAML or dictionary-based configuration system
- **Wavelength Grids**: Customizable wavelength bins for spectral analysis

## Documentation

📚 **[Full Documentation Site](https://meighenbergers.github.io/fennel/)** (MkDocs)

### Quick Links
- **Getting Started**
  - [Installation Guide](https://meighenbergers.github.io/fennel/getting-started/installation/) - PyPI, GitHub, and development setups
  - [Quickstart Tutorial](https://meighenbergers.github.io/fennel/getting-started/quickstart/) - Your first calculations
  - [Basic Examples](https://meighenbergers.github.io/fennel/getting-started/examples/) - Common use cases

- **User Guides**
  - [Configuration](https://meighenbergers.github.io/fennel/user-guide/configuration/) - YAML and dictionary setup
  - [Muon Tracks](https://meighenbergers.github.io/fennel/user-guide/tracks/) - Track yield calculations
  - [EM Cascades](https://meighenbergers.github.io/fennel/user-guide/em-cascades/) - Electromagnetic showers
  - [Hadron Cascades](https://meighenbergers.github.io/fennel/user-guide/hadron-cascades/) - Hadronic showers
  - [Advanced Usage](https://meighenbergers.github.io/fennel/user-guide/advanced/) - JAX, batching, and optimization

- **API Reference**
  - [Fennel Core API](https://meighenbergers.github.io/fennel/api/fennel/) - Main interface
  - [Results Containers](https://meighenbergers.github.io/fennel/api/results/) - v2 result objects
  - [Configuration](https://meighenbergers.github.io/fennel/api/config/) - Config management

- **Development**
  - [Contributor Guide](CONTRIBUTING.md) - How to contribute
  - [PR Guide](https://meighenbergers.github.io/fennel/development/pr-guide/) - PR guidelines and workflow
  - [Upgrade Guide](https://meighenbergers.github.io/fennel/getting-started/upgrade-guide/) - Migration from v1 to v2
  - [Changelog](CHANGELOG.md) - Version history

### Local Documentation
- MkDocs sources: [docs-mkdocs/](docs-mkdocs/) directory
- Build locally: `mkdocs serve` (see [docs-mkdocs/README.md](docs-mkdocs/README.md))

## Installation

### ⚠️ Important: Version Differences

**PyPI (v1.3.4)**: Stable release with classic tuple-based API  
**GitHub (v2.0.0)**: Latest release with new structured API and convenience methods

> **PyPI is currently on v1.3.4** and uses the classic tuple API. For the new v2.0.0 API with structured results, validation, and convenience methods, install from GitHub until the PyPI package is updated.

### PyPI - Stable v1.3.4 (Classic API)
```bash
pip install fennel_seed
```
This version uses tuple returns and is production-stable. Perfect if you need stability or have existing v1 code.

### GitHub - Latest v2.0.0 (Recommended for New Projects)

**⭐ Recommended for new projects** - includes all v2 features while remaining 100% backward compatible.
```bash
# Base install
pip install "fennel_seed @ git+https://github.com/MeighenBergerS/fennel.git@master"

# With JAX for GPU acceleration
pip install "fennel_seed[jax] @ git+https://github.com/MeighenBergerS/fennel.git@master"

# With interactive/notebook features
pip install "fennel_seed[interactive] @ git+https://github.com/MeighenBergerS/fennel.git@master"
```

📖 **More installation options**: [Installation Guide](https://meighenbergers.github.io/fennel/getting-started/installation/)

### Development Install (From Source)
```bash
git clone https://github.com/MeighenBergerS/fennel.git
cd fennel
python3 -m venv .venv && source .venv/bin/activate

# Editable install
pip install -e .

# With development tools (testing, docs, linting)
pip install -e .[dev]
```

Verify installation:
```bash
python -c "import fennel; print(f'Fennel {fennel.__version__} ready!')"
```

Quick functionality check:
```python
from fennel import Fennel
f = Fennel()

# v2 API
result = f.quick_track(energy=100.0, interaction="total")
print(f"Track result: {result.photons.sum():.2e} photons")

# v1 API (still works)
_, photons = f.track_yields(energy=100.0, particle=13)
print(f"Legacy API: {photons.sum():.2e} photons")

f.close()
```

## Quickstart (v2 API)

### Basic Usage

```python
from fennel import Fennel

f = Fennel()

# 🎯 Quick track calculation (100 GeV muon)
track = f.quick_track(energy=100.0, interaction="total")
print(f"Energy: {track.energy} GeV")
print(f"Particle: {track.particle_name}")
print(f"Total photons: {track.photons.sum():.2e}")
print(f"Yield shape: {track.dcounts.shape}")

# 🎯 Quick cascade (automatically detects EM vs hadron)
cascade = f.quick_cascade(energy=1000.0, particle=11)  # electron
print(f"Cascade from {cascade.particle_name}")
print(f"Profile: {cascade.longitudinal_profile.shape}")

# 🎯 Universal calculator (auto-detects everything)
result = f.calculate(energy=500.0, particle=211)  # π+ (hadron)
print(f"Type: {result.interaction}")
if hasattr(result, 'em_fraction'):
    print(f"EM fraction: {result.em_fraction:.2%}")

f.close()
```

### v2 Features

```python
# Structured results with named attributes
track = f.track_yields_v2(energy=100.0, particle=13, interaction="total")
print(track.wavelengths)  # Named attribute instead of tuple unpacking
print(track.photons)
print(track.dcounts)

# Pretty printing
print(track)  # Shows all contained data

# Legacy v1 API (100% compatible)
wavelengths, photons = f.track_yields(energy=100.0, particle=13)
print(f"v1 API: {photons.sum():.2e} photons")
```

### More Examples
- 📘 [Full Quickstart Tutorial](https://meighenbergers.github.io/fennel/getting-started/quickstart/)
- 📓 [Example Notebooks](notebooks/) - `example_v2.ipynb`, `example.ipynb`
- 📖 [User Guides](https://meighenbergers.github.io/fennel/user-guide/configuration/) - Detailed usage for each feature

## Testing

Fennel has comprehensive test coverage (123+ tests) covering API, validation, physics, and backward compatibility.

```bash
# Run all tests
pytest tests/

# Specific test suites
pytest tests/test_v2_api.py              # v2 API features
pytest tests/test_physics_regression.py  # Physics validation
pytest tests/test_integration.py         # End-to-end tests

# With coverage report
pytest --cov=fennel tests/
```

See [TEST_SETUP_COMPLETE.md](TEST_SETUP_COMPLETE.md) for test infrastructure details.

## Development

### Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- How to set up your development environment
- Code style and testing guidelines
- Pull request process
- Commit conventions

Also see:
- [Testing Guide](https://meighenbergers.github.io/fennel/development/testing/) - Running and writing tests
- [PR Guide](https://meighenbergers.github.io/fennel/development/pr-guide/) - Pull request workflow
- [Commit Guide](https://meighenbergers.github.io/fennel/development/commit-guide/) - Commit conventions

### Pre-commit Hooks

This repository uses pre-commit hooks for code quality and notebook cleanliness.

```bash
# Install hooks
pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files

# Specific hooks
pre-commit run nbstripout --all-files  # Clean notebook outputs
```

### Documentation

Build and serve documentation locally:

```bash
# Install MkDocs
pip install -r docs-mkdocs/requirements.txt

# Serve locally (with live reload)
mkdocs serve

# Build static site
mkdocs build
```

Documentation source: [docs-mkdocs/](docs-mkdocs/)

## Citation

If you use Fennel in your research, please cite:

### Software Citation

```bibtex
@software{fennel2024,
  author = {Meighen-Berger, Stephan},
  title = {Fennel: Cherenkov Light Yield Simulation},
  url = {https://github.com/MeighenBergerS/fennel},
  version = {2.0.0},
  year = {2024}
}
```

### Parametrization References

Fennel uses the Aachen parametrization based on GEANT4 simulations. Please also cite:

**Rädel, L., & Wiebusch, C.** (2012). *Calculation of the Cherenkov light yield from low energetic secondary particles accompanying high-energy muons in ice and water with Geant4 simulations*. Astroparticle Physics, 38, 53-67. [doi:10.1016/j.astropartphys.2012.09.008](https://doi.org/10.1016/j.astropartphys.2012.09.008)

Additional references: [Citation Guide](https://meighenbergers.github.io/fennel/about/citation/)

## Support

- **📖 Documentation**: [https://meighenbergers.github.io/fennel/](https://meighenbergers.github.io/fennel/)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/MeighenBergerS/fennel/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/MeighenBergerS/fennel/discussions)
- **📧 Contact**: Open an issue or discussion for questions

### Troubleshooting

Common issues and solutions:

- **Import errors**: Ensure you have Python 3.8+ and all dependencies installed
- **Version confusion**: Check `fennel.__version__` to verify which version you have
- **JAX issues**: JAX is optional; install with `pip install fennel_seed[jax]`
- **Validation errors**: v2 API provides detailed error messages - read them carefully!

For more help, see the [documentation](https://meighenbergers.github.io/fennel/) or open an issue.

## Beta Projects

Fennel includes experimental subprojects available in the repository:

### Seed (GEANT4 Interface)

A GEANT4-based interface for generating and validating parametrizations.

- **Location**: [seed/](seed/) directory
- **Requirements**: GEANT4 installation (tested on Linux)
- **Documentation**: See [seed/README.md](seed/README.md)
- **Examples**: Included in [seed/examples/](seed/examples/)

⚠️ **Beta Status**: This component is experimental and primarily for advanced users interested in parametrization development.

---

**License**: MIT - See [LICENSE.txt](LICENSE.txt) for details
