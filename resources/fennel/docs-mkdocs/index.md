# Fennel

<p align="center">
  <img src="assets/Fennel.png" alt="Fennel Logo" width="300"/>
</p>

**Cherenkov light yield simulation for particles and cascades**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)

Fennel is a Python package for simulating Cherenkov light production from particles. It calculates light emissions from both cascades (electromagnetic and hadronic) and tracks using the Aachen parametrization.

---

## Features

✨ **Comprehensive Physics Coverage**

- **Tracks**: Muon light yield with energy loss mechanisms (ionization, bremsstrahlung, pair production)
- **EM Cascades**: Electron, positron, and photon-initiated showers
- **Hadron Cascades**: Pion, kaon, proton, and neutron-initiated showers with EM fraction and muon production

🚀 **High Performance**

- NumPy-based calculations for CPU efficiency
- Optional JAX backend for GPU acceleration
- Optimized parametrizations from GEANT4 simulations

🎯 **User-Friendly API**

- Simple, intuitive interface
- Comprehensive type hints for IDE support
- Detailed NumPy-style documentation
- Flexible configuration via YAML or dictionaries

📊 **Wavelength-Dependent Yields**

- Customizable wavelength grids
- Angular distributions
- Longitudinal shower profiles

---

## Quick Example

```python
from fennel import Fennel

# Initialize
f = Fennel()

# Calculate muon track yields
wavelengths, photons = f.track_yields(
    energy=100.0,  # 100 GeV muon
    particle=13,    # PDG code for muon
    wavelength=400  # 400 nm
)

# Calculate EM cascade yields
wavelengths, total_photons, profile, angles = f.em_yields(
    energy=100.0,   # 100 GeV
    particle=11     # electron
)

# Calculate hadron cascade yields  
results = f.hadron_yields(
    energy=100.0,   # 100 GeV
    particle=2212   # proton
)
```

---

## Installation

```bash
pip install fennel_seed
```

For JAX GPU acceleration:
```bash
pip install fennel_seed[jax]
```

For development:
```bash
git clone https://github.com/MeighenBergerS/fennel.git
cd fennel
pip install -e .[dev]
```

---

## Key Concepts

### Tracks vs Cascades

- **Tracks**: Charged particles (primarily muons) that travel long distances while emitting Cherenkov light continuously
- **EM Cascades**: Electromagnetic showers from electrons, positrons, or photons that develop through pair production and bremsstrahlung
- **Hadron Cascades**: Hadronic showers from pions, kaons, protons, or neutrons with both electromagnetic and purely hadronic components

### Parametrization

Fennel uses the **Aachen parametrization** based on detailed GEANT4 simulations. The parametrization includes:

- Energy-dependent track lengths and shower profiles
- Angular distributions of Cherenkov photons
- Wavelength-dependent yields
- EM fraction in hadronic showers
- Muon production in hadronic interactions

---

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and first steps
- **[User Guide](user-guide/configuration.md)**: Detailed usage instructions
- **[API Reference](api/fennel.md)**: Complete API documentation
- **[Physics Background](physics/cherenkov.md)**: Theory and parametrizations
- **[Development](development/contributing.md)**: Contributing and testing

---

## Citation

If you use Fennel in your research, please cite:

```bibtex
@software{fennel2024,
  author = {Meighen-Berger, Stephan},
  title = {Fennel: Cherenkov Light Yield Simulation},
  year = {2024},
  url = {https://github.com/MeighenBergerS/fennel},
  version = {1.3.4}
}
```

The package includes distributions developed in multiple studies - see [Citation](about/citation.md) for complete references.

---

## License

Fennel is released under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/MeighenBergerS/fennel/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeighenBergerS/fennel/discussions)

---

## Acknowledgments

Developed by Stephan Meighen-Berger based on the Aachen parametrization for Cherenkov light production.
