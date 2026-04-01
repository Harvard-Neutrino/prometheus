# Installation

## Requirements

- Python 3.8 or higher
- NumPy
- SciPy
- PyYAML
- pandas

## Install from PyPI

The easiest way to install Fennel is via pip:

```bash
pip install fennel_seed
```

This will install Fennel with all required dependencies.

## Optional Dependencies

### JAX for GPU Acceleration

For GPU-accelerated calculations using JAX:

```bash
pip install fennel_seed[jax]
# or
pip install "jax[cuda12]"  # for CUDA 12
pip install "jax[cuda11]"  # for CUDA 11
```

!!! note "JAX Performance"
    JAX provides significant speedups for large batch calculations and is recommended if you have GPU access. The API remains the same whether using NumPy or JAX.

## Install from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/MeighenBergerS/fennel.git
cd fennel

# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e .[dev]
```

### Development Dependencies

The `[dev]` extras include:

- `pytest` - for running tests
- `pytest-cov` - for coverage reports
- `mkdocs` and plugins - for building documentation

## Verify Installation

Test your installation:

```python
import fennel
print(fennel.__version__)

# Quick functionality test
from fennel import Fennel
f = Fennel()
wavelengths, photons = f.track_yields(energy=100.0, particle=13)
print(f"Success! Generated {photons.sum():.2e} photons")
```

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Ensure you have the latest pip
pip install --upgrade pip

# Reinstall fennel
pip install --force-reinstall fennel_seed
```

### JAX Issues

If JAX installation fails:

```bash
# Install CPU-only JAX first
pip install jax

# Then install fennel
pip install fennel_seed
```

For detailed JAX installation instructions, see the [JAX documentation](https://github.com/google/jax#installation).

## Configuration

After installation, you can customize Fennel's behavior. See the [Configuration](../user-guide/configuration.md) guide for details.

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Get started with basic examples
- [Configuration Guide](../user-guide/configuration.md) - Learn about customization options
- [API Reference](../api/fennel.md) - Explore the complete API
