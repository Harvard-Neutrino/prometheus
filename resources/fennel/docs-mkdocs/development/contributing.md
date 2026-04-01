# Contributing to Fennel

Thank you for your interest in contributing to Fennel! This guide will help you get started.

## Quick Start

1. **Fork and clone** the repository
2. **Set up** your development environment
3. **Make changes** in a feature branch
4. **Test** your changes
5. **Submit** a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fennel.git
cd fennel

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Pre-commit Hooks

Fennel uses pre-commit hooks to maintain code quality:

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run nbstripout --all-files
```

The hooks will:
- Format code with Black
- Sort imports with isort
- Strip notebook outputs
- Check for common issues

## Making Changes

### Branching Strategy

- `master` - Main development branch
- `feature/your-feature` - For new features
- `fix/your-fix` - For bug fixes
- `docs/your-docs` - For documentation updates

```bash
# Create a feature branch
git checkout -b feature/my-new-feature
```

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings in NumPy style
- Keep functions focused and small

### Testing

Run tests before committing:

```bash
# All tests
pytest

# Fast tests only
pytest -m "not slow"

# Specific test file
pytest tests/test_v2_api.py

# With coverage
pytest --cov=fennel
```

See [Testing Guide](testing.md) for more details.

### Documentation

Update documentation when adding features:

```bash
# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

Documentation lives in `docs-mkdocs/`. Write in Markdown with:
- Clear examples
- Code snippets
- Cross-references

## Pull Request Process

### Before Submitting

1. ✅ **Tests pass**: `pytest`
2. ✅ **Code formatted**: `pre-commit run --all-files`
3. ✅ **Docs updated**: If adding features
4. ✅ **CHANGELOG updated**: Add entry under `[Unreleased]`

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests (if applicable)

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Automated tests run via GitHub Actions
2. Maintainer reviews code
3. Address feedback
4. Maintainer merges PR

## Commit Guidelines

Write clear commit messages:

```bash
# Good
git commit -m "Add validation for negative energies"
git commit -m "Fix: Handle zero wavelength edge case"
git commit -m "Docs: Update installation guide"

# Bad
git commit -m "fix stuff"
git commit -m "WIP"
```

See [Commit Guide](commit-guide.md) for detailed conventions.

## CHANGELOG Conventions

Add entries under `[Unreleased]` section:

```markdown
### Added
- New feature description

### Changed
- Modified behavior description

### Fixed
- Bug fix description

### Deprecated
- Deprecated feature description
```

## Questions or Issues?

- **Bug reports**: [GitHub Issues](https://github.com/MeighenBergerS/fennel/issues)
- **Questions**: [GitHub Discussions](https://github.com/MeighenBergerS/fennel/discussions)
- **Security issues**: Email maintainer privately

## Code of Conduct

Be respectful and constructive. We're all here to improve Fennel together!

---

Thank you for contributing to Fennel! 🌿
