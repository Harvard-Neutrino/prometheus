# Quick Start

This guide will get you up and running with Fennel in minutes.

## Basic Usage

### 1. Import and Initialize

```python
from fennel import Fennel

# Initialize with default configuration
f = Fennel()
```

### 2. Calculate Track Yields (Muons)

Calculate Cherenkov light from a muon track:

```python
# 100 GeV muon at 400 nm
wavelengths, photons = f.track_yields(
    energy=100.0,      # GeV
    particle=13,       # PDG code for μ⁻
    wavelength=400     # nm
)

print(f"Photons per meter: {photons * 100:.2e}")
```

Calculate wavelength spectrum:

```python
import numpy as np

# Define wavelength range
wavelengths = np.linspace(300, 600, 100)  # 300-600 nm

# Get spectrum
_, photons = f.track_yields(
    energy=100.0,
    particle=13,
    wavelength=wavelengths
)

# Plot
import matplotlib.pyplot as plt
plt.plot(wavelengths, photons * 100)  # photons/m
plt.xlabel('Wavelength [nm]')
plt.ylabel('dN/dλ per meter [1/(nm·m)]')
plt.title('100 GeV Muon Cherenkov Spectrum')
plt.show()
```

### 3. Calculate EM Cascade Yields

For electromagnetic showers:

```python
# 100 GeV electron
wavelengths, total_photons, profile, angles = f.em_yields(
    energy=100.0,
    particle=11,       # PDG code for e⁻
    wavelength=400
)

print(f"Total photons: {total_photons:.2e}")
```

### 4. Calculate Hadron Cascade Yields

For hadronic showers:

```python
# 100 GeV proton
(wavelengths, total_photons, em_fraction, 
 em_fraction_std, profile, angles) = f.hadron_yields(
    energy=100.0,
    particle=2212      # PDG code for proton
)

print(f"Total photons: {total_photons:.2e}")
print(f"EM fraction: {em_fraction:.3f} ± {em_fraction_std:.3f}")
```

## Common Particle Codes (PDG)

| Particle | PDG Code |
|----------|----------|
| μ⁻ (muon) | 13 |
| μ⁺ (anti-muon) | -13 |
| e⁻ (electron) | 11 |
| e⁺ (positron) | -11 |
| γ (photon) | 22 |
| π⁺ (pion plus) | 211 |
| π⁻ (pion minus) | -211 |
| K⁺ (kaon plus) | 321 |
| p (proton) | 2212 |
| n (neutron) | 2112 |

## Energy Loss Mechanisms

For tracks, you can specify different energy loss mechanisms:

```python
# Total light yield (default)
_, photons_total = f.track_yields(
    energy=100.0, particle=13, interaction='total'
)

# Only ionization losses
_, photons_ion = f.track_yields(
    energy=100.0, particle=13, interaction='ionization'
)

# Only bremsstrahlung
_, photons_brems = f.track_yields(
    energy=100.0, particle=13, interaction='brems'
)

# Only pair production
_, photons_pair = f.track_yields(
    energy=100.0, particle=13, interaction='pair'
)
```

## Working with Arrays

Calculate yields for multiple energies:

```python
energies = np.logspace(0, 3, 50)  # 1 GeV to 1 TeV

photon_yields = []
for energy in energies:
    _, photons = f.track_yields(
        energy=energy,
        particle=13,
        wavelength=400
    )
    photon_yields.append(photons)

# Plot energy dependence
plt.loglog(energies, photon_yields)
plt.xlabel('Energy [GeV]')
plt.ylabel('Photons per meter')
plt.title('Muon Light Yield vs Energy')
plt.grid(True, alpha=0.3)
plt.show()
```

## Custom Configuration

Use a custom configuration:

```python
# From dictionary
config = {
    'scenario': {
        'medium': 'ice',
        'parametrization': 'aachen'
    },
    'advanced': {
        'wavelengths': [300, 400, 500, 600]  # nm
    }
}

f = Fennel(userconfig=config)
```

Or from a YAML file:

```python
# Load from YAML
f = Fennel(userconfig='path/to/config.yaml')
```

## Angular Distributions

Get angular emission patterns:

```python
# Track angular distribution
wavelengths, photons, angles, angle_dist = f.track_yields(
    energy=100.0,
    particle=13,
    wavelength=400,
    angles=True  # Request angular distribution
)

# Plot
import numpy as np
angle_grid = np.linspace(0, 10, 100)  # degrees
plt.plot(angle_grid, angle_dist)
plt.xlabel('Angle [degrees]')
plt.ylabel('Photons per degree')
plt.title('Cherenkov Angular Distribution')
plt.show()
```

## Longitudinal Profiles

For cascades, examine the shower development:

```python
# Get longitudinal profile
import numpy as np

z_grid = np.linspace(0, 1000, 100)  # 0-1000 cm

_, _, profile, _ = f.em_yields(
    energy=100.0,
    particle=11,
    z_grid=z_grid
)

plt.plot(z_grid, profile)
plt.xlabel('Depth [cm]')
plt.ylabel('Light production [1/cm]')
plt.title('EM Shower Development')
plt.show()
```

## Next Steps

- [Configuration Guide](../user-guide/configuration.md) - Customize Fennel behavior
- [Basic Examples](examples.md) - More detailed examples
- [API Reference](../api/fennel.md) - Complete API documentation
- [Physics Background](../physics/cherenkov.md) - Understand the physics

## Performance Tips

!!! tip "Batch Calculations"
    For multiple calculations, reuse the `Fennel` object instead of creating new ones:
    ```python
    f = Fennel()  # Create once
    
    # Use many times
    for energy in energies:
        result = f.track_yields(energy=energy, particle=13)
    ```

!!! tip "JAX Acceleration"
    For large batch calculations, enable JAX:
    ```python
    config = {'general': {'jax': True}}
    f = Fennel(userconfig=config)
    ```
