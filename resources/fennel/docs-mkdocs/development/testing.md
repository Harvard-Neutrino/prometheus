# Testing Guide

Fennel has a comprehensive test suite covering API functionality, validation, physics calculations, and backward compatibility.

## Quick Start

```bash
# Run all tests
pytest

# Fast tests only (skip slow tests)
pytest -m "not slow"

# With verbose output
pytest -v

# With coverage report
pytest --cov=fennel --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_config.py           # Configuration tests
├── test_em_cascades.py      # EM cascade tests
├── test_hadron_cascades.py  # Hadron cascade tests
├── test_tracks.py           # Track yield tests
├── test_v2_api.py          # v2 API tests
├── test_integration.py      # Integration tests
└── test_physics_regression.py  # Physics validation
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```bash
pytest -m unit
```

### Integration Tests

Test component interactions:

```bash
pytest -m integration
```

### Physics Regression Tests

Validate physics calculations against reference values:

```bash
pytest tests/test_physics_regression.py -v
```

These tests ensure:
- Photon yields match expected values
- Longitudinal profiles are correct
- Angular distributions are accurate

### v2 API Tests

Test new v2 API features:

```bash
pytest tests/test_v2_api.py -v
```

Coverage includes:
- Result container classes
- Convenience methods (`quick_track`, `quick_cascade`, `calculate`)
- Input validation
- Backward compatibility

## Markers

Tests are marked for selective execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.physics` - Physics validation
- `@pytest.mark.slow` - Slow tests (skip in fast mode)
- `@pytest.mark.jax` - JAX-specific tests (requires JAX)

Example:

```python
@pytest.mark.physics
@pytest.mark.slow
def test_detailed_physics():
    # Long-running physics test
    pass
```

## Using Make Commands

The Makefile provides convenient test commands:

```bash
make test              # All tests
make test-fast         # Skip slow tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-physics      # Physics tests only
make test-cov          # Tests with coverage report
```

## Writing Tests

### Test Fixtures

Use pytest fixtures for common setup:

```python
import pytest
from fennel import Fennel

@pytest.fixture
def fennel_instance():
    f = Fennel()
    yield f
    f.close()

def test_something(fennel_instance):
    result = fennel_instance.quick_track(energy=100.0)
    assert result.energy == 100.0
```

### Test Structure

Follow AAA pattern (Arrange, Act, Assert):

```python
def test_track_yields_v2():
    # Arrange
    f = Fennel()
    energy = 100.0
    
    # Act
    result = f.track_yields_v2(energy=energy, particle=13)
    
    # Assert
    assert result.energy == energy
    assert result.particle_name == "μ⁻"
    assert result.photons.shape[0] > 0
    
    f.close()
```

### Testing Validation

Test that validation catches bad inputs:

```python
import pytest
from fennel import Fennel, ValidationError

def test_negative_energy_raises():
    f = Fennel()
    
    with pytest.raises(ValidationError, match="Energy must be positive"):
        f.quick_track(energy=-100.0)
    
    f.close()
```

### Testing Edge Cases

Always test boundary conditions:

```python
def test_zero_energy():
    # Test behavior at boundaries
    pass

def test_extreme_energy():
    # Test at extreme values
    pass

def test_invalid_particle():
    # Test error handling
    pass
```

## Coverage Requirements

- Aim for >80% coverage on new code
- Critical paths should have 100% coverage
- Test both success and failure cases

View coverage report:

```bash
pytest --cov=fennel --cov-report=html
# Open htmlcov/index.html in browser
```

## Continuous Integration

GitHub Actions runs tests automatically on:
- Every push
- Every pull request
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Multiple platforms (Ubuntu, macOS, Windows)

## Reference Values

Physics tests compare against reference values in:
- `tests/reference_values_v1.3.4.json`

To regenerate reference values:

```bash
python scripts/generate_reference_values.py
```

⚠️ Only regenerate if you've intentionally changed physics calculations!

## Debugging Tests

### Run specific test

```bash
pytest tests/test_v2_api.py::test_track_yields_v2
```

### Show print statements

```bash
pytest -s
```

### Drop into debugger on failure

```bash
pytest --pdb
```

### Verbose output

```bash
pytest -vv
```

## Performance Testing

For performance-critical code:

```bash
pytest --durations=10  # Show 10 slowest tests
```

## Best Practices

1. ✅ Test one thing per test function
2. ✅ Use descriptive test names
3. ✅ Keep tests independent
4. ✅ Use fixtures for setup
5. ✅ Test edge cases
6. ✅ Test error handling
7. ✅ Keep tests fast (mark slow tests)
8. ✅ Don't test implementation details

---

See also: [PR Guide](pr-guide.md) | [Contributing](contributing.md)
