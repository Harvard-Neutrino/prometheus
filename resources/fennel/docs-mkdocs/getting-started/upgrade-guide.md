# Fennel v2.0 Upgrade Guide

## Overview

Fennel v2.0 introduces significant API improvements while maintaining **100% backward compatibility**. All existing code will continue to work without modifications. This guide shows you how to take advantage of the new features.

## What's New in v2.0

### 1. Result Container Classes

Instead of remembering tuple unpacking order, use structured result objects:

**Old way (still works):**
```python
dcounts, angles = fennel.track_yields(100.0)
dcounts, dcounts_s, long_prof, angles = fennel.em_yields(1000.0, particle=11)
```

**New way:**
```python
result = fennel.track_yields_v2(100.0)
print(result.dcounts)
print(result.angles)
print(result.energy)  # Metadata included!
print(result)  # Pretty print: TrackYieldResult(energy=100.0 GeV, ...)

em_result = fennel.em_yields_v2(1000.0, particle=11)
print(em_result.particle_name)  # "electron"
print(em_result.dcounts)
print(em_result.longitudinal_profile)
```

### 2. Better Error Messages with Validation

**Old behavior:**
```python
fennel.track_yields(-100.0)  # Cryptic error deep in the code
```

**New behavior:**
```python
from fennel import ValidationError

try:
    fennel.track_yields_v2(-100.0)
except ValidationError as e:
    print(e)
    # "Energy must be positive, got -100.0 GeV. Example: energy=100.0"
```

**Helpful suggestions for common mistakes:**
```python
fennel.em_yields_v2(1000.0, particle=999)
# ValidationError: Unknown particle PDG ID: 999. 
# Supported particles: [11, -11, 22, ...].
# Common examples:
#   EM cascades: 11 (electron), -11 (positron), 22 (photon)
```

### 3. Convenience Methods

**Quick calculations with minimal parameters:**
```python
# Old way
dcounts, angles = fennel.track_yields(
    100.0,
    wavelengths=config["advanced"]["wavelengths"],
    angle_grid=config["advanced"]["angles"],
    n=config["mediums"][config["scenario"]["medium"]]["refractive index"]
)

# New way
result = fennel.quick_track(100.0)
```

**Universal `calculate()` method:**
```python
# Auto-detects particle type from PDG ID
result = fennel.calculate(100.0, particle=11)      # EM cascade
result = fennel.calculate(100.0, particle=211)     # Hadron cascade
result = fennel.calculate(100.0, particle=13)      # Track

# Or use friendly names
result = fennel.calculate(100.0, particle_type='electron')
result = fennel.calculate(100.0, particle_type='pion')
result = fennel.calculate(100.0, particle_type='muon')
```

**Quick cascade for any particle:**
```python
# Automatically routes to em_yields or hadron_yields
electron_result = fennel.quick_cascade(1000.0, particle=11)
pion_result = fennel.quick_cascade(1000.0, particle=211)
```

## Migration Examples

### Example 1: Track Analysis

**Before (v1.x):**
```python
from fennel import Fennel
import numpy as np

fennel = Fennel()
wavelengths = np.linspace(300, 600, 100)

# Must remember tuple order
dcounts, angles = fennel.track_yields(100.0, wavelengths=wavelengths)

# Calculate total photons
total_photons = np.trapezoid(dcounts, wavelengths)
print(f"Total photons: {total_photons}")

fennel.close()
```

**After (v2.0 - recommended):**
```python
from fennel import Fennel
import numpy as np

fennel = Fennel()

# Structured result with metadata
result = fennel.track_yields_v2(100.0)

# Self-documenting code
total_photons = np.trapezoid(result.dcounts, wavelengths)
print(f"Energy: {result.energy} GeV")
print(f"Interaction: {result.interaction}")
print(f"Total photons: {total_photons}")
print(result)  # Pretty print

fennel.close()
```

### Example 2: Cascade Comparison

**Before (v1.x):**
```python
# EM cascade
em_dc, em_dcs, em_lp, em_ang = fennel.em_yields(1000.0, particle=11)

# Hadron cascade - different tuple size!
had_dc, had_dcs, had_em_frac, had_em_frac_s, had_lp, had_ang = \
    fennel.hadron_yields(1000.0, particle=211)

print(f"Hadron EM fraction: {had_em_frac:.2%}")
```

**After (v2.0 - recommended):**
```python
# Consistent interface
em_result = fennel.em_yields_v2(1000.0, particle=11)
hadron_result = fennel.hadron_yields_v2(1000.0, particle=211)

print(f"EM particle: {em_result.particle_name}")
print(f"Hadron particle: {hadron_result.particle_name}")
print(f"Hadron EM fraction: {hadron_result.em_fraction:.2%}")

# Or use convenience method
electron = fennel.quick_cascade(1000.0, particle=11)
pion = fennel.quick_cascade(1000.0, particle=211)
```

### Example 3: Error Handling

**Before (v1.x):**
```python
# Errors could be cryptic
try:
    dcounts, angles = fennel.track_yields(-100.0)
except Exception as e:
    print(f"Something went wrong: {e}")
    # User has to debug what went wrong
```

**After (v2.0 - recommended):**
```python
from fennel import ValidationError

try:
    result = fennel.track_yields_v2(-100.0)
except ValidationError as e:
    print(e)
    # Clear message: "Energy must be positive, got -100.0 GeV. Example: energy=100.0"
    # User knows exactly what to fix!
```

## API Comparison Table

| Feature | Old API | New API (v2.0) | Status |
|---------|---------|----------------|--------|
| Track yields | `track_yields()` | `track_yields_v2()` | Both work |
| EM cascade yields | `em_yields()` | `em_yields_v2()` | Both work |
| Hadron cascade yields | `hadron_yields()` | `hadron_yields_v2()` | Both work |
| Return type | Tuple | Result Container | v2 only |
| Input validation | Minimal | Comprehensive | v2 only |
| Error messages | Basic | Helpful with examples | v2 only |
| Quick methods | - | `quick_track()`, `quick_cascade()` | v2 only |
| Universal method | `auto_yields()` | `calculate()` | Both work |

## When to Upgrade

### Use v2 API when:
- Starting a new project
- You want better error messages
- You prefer self-documenting code
- You're tired of remembering tuple order
- You want IDE autocomplete for result attributes

### Keep old API when:
- You have working code you don't want to touch
- You're maintaining legacy code
- You prefer the terseness of tuple unpacking

## Backward Compatibility Guarantee

**All v1.x code works in v2.0** without any changes:

```python
# This still works exactly as before
from fennel import Fennel, config

config["general"]["random state seed"] = 42
fennel = Fennel()

# Old API - still supported
dcounts, angles = fennel.track_yields(100.0)
dcounts, dcounts_s, long_prof, angles = fennel.em_yields(1000.0, particle=11)
dcounts, dcounts_s, em_frac, em_frac_s, long_prof, angles = \
    fennel.hadron_yields(1000.0, particle=211)

fennel.close()
```

## New Exports

```python
from fennel import (
    Fennel,                  # Main class (same as before)
    config,                  # Configuration (same as before)
    TrackYieldResult,        # New: Result container for tracks
    EMYieldResult,           # New: Result container for EM cascades
    HadronYieldResult,       # New: Result container for hadron cascades
    ValidationError          # New: Clear error messages
)
```

## Testing

All 40 physics regression tests pass - calculations are identical to v1.x.
20 new tests verify v2 API features and backward compatibility.

```bash
# Run all tests
pytest tests/

# Run regression tests (verify physics unchanged)
pytest tests/test_physics_regression.py

# Run v2 API tests
pytest tests/test_v2_api.py
```

## Performance

No performance impact - the v2 API is a thin wrapper around the existing implementation. The old and new APIs produce identical results.

## Questions?

- **Q: Do I need to upgrade my code?**
  - A: No! All existing code works without changes.

- **Q: What if I find the new API too verbose?**
  - A: Keep using the old API! It's fully supported.

- **Q: Can I mix old and new APIs?**
  - A: Absolutely! Use whichever is most convenient for each situation.

- **Q: Will the old API be deprecated?**
  - A: No plans to deprecate. Both APIs are first-class citizens.

## Summary

Fennel v2.0 offers significant quality-of-life improvements while respecting your existing codebase. Adopt the new features at your own pace - there's no pressure to change working code. The choice is yours!
