# Configuration

Fennel can be configured in three ways:

1. **Default configuration** - Use built-in defaults
2. **Dictionary configuration** - Pass a Python dictionary
3. **YAML configuration** - Load from a YAML file

## Default Configuration

The simplest approach:

```python
from fennel import Fennel

f = Fennel()  # Uses all defaults
```

## Dictionary Configuration

Customize specific settings:

```python
config = {
    'scenario': {
        'medium': 'ice',  # or 'water'
        'parametrization': 'aachen'
    },
    'general': {
        'jax': False,  # Set to True for GPU acceleration
        'enable logging': True
    },
    'advanced': {
        'wavelengths': [300, 350, 400, 450, 500, 550, 600]  # nm
    }
}

f = Fennel(userconfig=config)
```

## YAML Configuration

Create a `config.yaml` file:

```yaml
scenario:
  medium: ice
  parametrization: aachen

general:
  jax: false
  enable logging: true

advanced:
  wavelengths: [300, 400, 500, 600]
  angles: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

Load it:

```python
f = Fennel(userconfig='path/to/config.yaml')
```

## Configuration Options

### Scenario Settings

| Parameter | Options | Description |
|-----------|---------|-------------|
| `medium` | `'ice'`, `'water'` | Medium properties |
| `parametrization` | `'aachen'` | Parametrization to use |

### General Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jax` | bool | `False` | Enable JAX acceleration |
| `enable logging` | bool | `True` | Enable logging output |

### Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wavelengths` | list[float] | 300-600nm | Wavelength grid in nm |
| `angles` | list[float] | 0-10° | Angular grid in degrees |
| `z grid` | list[float] | Auto | Depth grid in cm |
| `track length` | float | 1.0 | Track segment length in cm |

## Medium Properties

### Ice (Default)

```python
config = {
    'scenario': {'medium': 'ice'},
    'mediums': {
        'ice': {
            'refractive index': 1.31,
            'radiation length': 36.08,  # g/cm²
            'density': 0.9216  # g/cm³
        }
    }
}
```

### Water

```python
config = {
    'scenario': {'medium': 'water'},
    'mediums': {
        'water': {
            'refractive index': 1.33,
            'radiation length': 36.08,  # g/cm²
            'density': 1.0  # g/cm³
        }
    }
}
```

## Particle Configuration

Specify which particles to simulate:

```python
config = {
    'simulation': {
        'track particles': [13, -13],  # μ⁻, μ⁺
        'em particles': [11, -11, 22],  # e⁻, e⁺, γ
        'hadron particles': [211, -211, 321, -321, 2212, -2212, 2112]
    }
}
```

## Example Configurations

### Minimal Configuration

```python
# Just change the medium
f = Fennel(userconfig={'scenario': {'medium': 'water'}})
```

### GPU-Accelerated

```python
# Enable JAX for GPU calculations
config = {'general': {'jax': True}}
f = Fennel(userconfig=config)
```

### Custom Wavelength Range

```python
import numpy as np

config = {
    'advanced': {
        'wavelengths': np.linspace(280, 700, 200).tolist()
    }
}
f = Fennel(userconfig=config)
```

## Accessing Configuration

View the active configuration:

```python
from fennel import config

print(config['scenario']['medium'])
print(config['mediums'][config['scenario']['medium']])
```

## Next Steps

- [Tracks Guide](tracks.md) - Working with track yields
- [EM Cascades Guide](em-cascades.md) - Electromagnetic showers
- [Advanced Usage](advanced.md) - Advanced features
