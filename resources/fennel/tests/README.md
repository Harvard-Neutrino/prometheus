# Testing Guide for Fennel

This directory contains comprehensive tests for the Fennel package to ensure code quality and physics consistency.

## Test Structure

```
tests/
├── __init__.py                      # Test package initialization
├── conftest.py                      # Pytest fixtures and configuration
├── test_tracks.py                   # Unit tests for Track class
├── test_em_cascades.py              # Unit tests for EM_Cascade class
├── test_hadron_cascades.py          # Unit tests for Hadron_Cascade class
├── test_integration.py              # Integration tests for Fennel API
├── test_physics_regression.py       # Physics regression tests
├── test_config.py                   # Configuration system tests
└── reference_values.json            # Gold standard physics values
```

## Running Tests

### Install Test Dependencies

```bash
# Install package with test dependencies
pip install -e .[test]

# Or install all development dependencies
pip install -e .[dev]
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only physics regression tests
pytest -m physics

# Run tests without JAX tests
pytest -m "not jax"

# Run fast tests only (skip slow tests)
pytest -m "not slow"
```

### Run Tests for Specific Components

```bash
# Test only tracks
pytest tests/test_tracks.py

# Test only EM cascades
pytest tests/test_em_cascades.py

# Test only hadron cascades
pytest tests/test_hadron_cascades.py

# Test only integration
pytest tests/test_integration.py
```

### Run with Coverage

```bash
pytest --cov=fennel --cov-report=html --cov-report=term
```

View coverage report:
```bash
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

### Run Tests in Parallel

```bash
pytest -n auto
```

## Physics Regression Tests

### Purpose

Physics regression tests ensure that the physical results produced by Fennel remain constant across code refactoring and improvements. These tests use "gold standard" reference values that must be preserved.

### Generating Reference Values

**IMPORTANT**: Reference values should be generated ONCE with the validated v1.3.4 code and committed to version control.

```bash
# Generate reference values
python scripts/generate_reference_values.py
```

This creates `tests/reference_values.json` with gold standard values for:
- Track yields (muon at 100 GeV and 10 TeV)
- EM cascade yields (electron, positron, gamma at 100 GeV)
- Hadron cascade yields (pions at 100 GeV, proton at 1 TeV)

### Running Regression Tests

```bash
pytest tests/test_physics_regression.py -v
```

If any regression test fails, it means the physics calculations have changed. This should only happen when:
1. There's a deliberate update to the physics parametrization
2. A bug has been fixed
3. **Never** due to code refactoring

### Updating Reference Values

Only update reference values when physics changes are intentional:

```bash
# Backup old values
cp tests/reference_values.json tests/reference_values.json.backup

# Generate new values
python scripts/generate_reference_values.py

# Verify changes are expected
git diff tests/reference_values.json

# Commit with clear explanation
git add tests/reference_values.json
git commit -m "Update reference values: <explanation of physics change>"
```

## Test Markers

Tests are marked with categories for selective running:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests for full workflows
- `@pytest.mark.physics` - Physics regression tests
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.jax` - Tests requiring JAX installation

## Writing New Tests

### Unit Test Example

```python
import pytest
from fennel.tracks import Track

@pytest.mark.unit
def test_track_functionality():
    """Test description."""
    track = Track()
    result = track.some_method(input_value)
    assert result == expected_value
```

### Integration Test Example

```python
@pytest.mark.integration
def test_full_workflow(fennel_instance):
    """Test description."""
    dcounts, angles = fennel_instance.track_yields(100.0)
    assert np.all(np.isfinite(dcounts))
```

### Physics Regression Test Example

```python
@pytest.mark.physics
def test_new_physics_value(reference_data):
    """Test description."""
    fennel = Fennel()
    result = fennel.calculate_something()
    expected = reference_data["expected_value"]
    assert np.abs(result - expected) < tolerance
```

## Continuous Integration

Tests run automatically on GitHub Actions for:
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Multiple operating systems (Ubuntu, macOS, Windows)
- With and without JAX
- Code coverage reporting

See `.github/workflows/tests.yml` for CI configuration.

## Test Philosophy

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test the public API and full workflows
3. **Physics Tests**: Ensure physics results never change unintentionally
4. **Fast by Default**: Most tests should run quickly
5. **Comprehensive Coverage**: Aim for >80% code coverage

## Troubleshooting

### Tests Fail After Code Changes

If physics regression tests fail:
1. Check if the physics actually changed (it shouldn't for refactoring)
2. Review your changes carefully
3. Run specific tests to identify the issue: `pytest tests/test_physics_regression.py::TestPhysicsRegression::test_specific_case -v`

### JAX Tests Failing

If JAX is not installed:
```bash
pip install jax jaxlib
# Or for CPU only
pip install -e .[cpu]
```

### Slow Tests

Skip slow tests during development:
```bash
pytest -m "not slow"
```

## Best Practices

1. **Run tests before committing**: `pytest -m "not slow"`
2. **Run full test suite before PR**: `pytest`
3. **Check coverage**: `pytest --cov=fennel`
4. **Never modify reference values** unless physics changes intentionally
5. **Add tests for new features**
6. **Add regression tests for bug fixes**

## Getting Help

If tests fail unexpectedly or you need help with testing:
1. Check the test output carefully
2. Run with verbose output: `pytest -vv`
3. Run a single test: `pytest tests/test_file.py::test_function_name -vv`
4. Check CI logs on GitHub
