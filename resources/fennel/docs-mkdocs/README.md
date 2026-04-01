# MkDocs Documentation (New)

This directory contains the new MkDocs Material documentation for Fennel.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

Or install from the main package:

```bash
pip install -e ".[docs]"
```

### Local Development

Serve the docs locally with live reload:

```bash
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

### Build Static Site

Build the documentation:

```bash
mkdocs build
```

The static site will be in the `site/` directory.

## Documentation Structure

```
docs-mkdocs/
├── index.md                  # Homepage
├── getting-started/
│   ├── installation.md       # Installation guide
│   ├── quickstart.md         # Quick start tutorial
│   └── examples.md           # Basic examples
├── user-guide/
│   ├── configuration.md      # Configuration options
│   ├── tracks.md            # Track yields guide
│   ├── em-cascades.md       # EM cascade guide
│   ├── hadron-cascades.md   # Hadron cascade guide
│   └── advanced.md          # Advanced features
├── api/
│   ├── fennel.md            # Main API reference
│   ├── config.md            # Configuration API
│   ├── particle.md          # Particle class
│   ├── tracks.md            # Track calculations
│   ├── em_cascades.md       # EM cascade calculations
│   ├── hadron_cascades.md   # Hadron cascade calculations
│   └── photons.md           # Photon yields
├── physics/
│   ├── cherenkov.md         # Cherenkov radiation basics
│   └── parametrizations.md # Parametrization details
└── about/
    ├── citation.md          # How to cite
    ├── license.md           # License information
    └── changelog.md         # Version history
```

## Contributing

When contributing to documentation:

1. Use clear, concise language
2. Include code examples
3. Add cross-references using `[text](../path/file.md)`
4. Test locally with `mkdocs serve`
5. Ensure all internal links work

## Features

This documentation uses:

- **Material Theme** - Modern, responsive design
- **mkdocstrings** - Auto-generated API docs from docstrings
- **MathJax** - LaTeX math rendering
- **Code highlighting** - Syntax highlighting for code blocks
- **Search** - Built-in search functionality
- **Dark mode** - Automatic dark/light theme switching

## Official Documentation

This is the **official documentation** for Fennel, automatically deployed to GitHub Pages.

### Deployment

- **URL**: https://meighenbergers.github.io/fennel/
- **Auto-deploy**: GitHub Actions deploys on every push to `master`
- **Workflow**: `.github/workflows/deploy-docs.yml`

### Previous Documentation

The old Sphinx-based documentation in `docs/` has been deprecated. See `docs/DEPRECATED.md` for details.

### Current Status

- ✅ Complete structure and content
- ✅ Auto-generated API docs
- ✅ Installation and quickstart
- ⏳ User guides (in progress)
- ⏳ Physics background (in progress)
- ⏳ Examples and tutorials (in progress)

## Questions?

For questions about the documentation:
- Open a [GitHub Issue](https://github.com/MeighenBergerS/fennel/issues)
- See the [Contributing Guide](development/contributing.md)
