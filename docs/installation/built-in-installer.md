# Install with Built-in Installer

The built-in installer sets up Prometheus and all its dependencies automatically. It works on Linux, macOS, and WSL2 — if you're on Windows (without WSL2), use [containers](containers.md) instead.

| Platform      | Status         |
| ------------- | -------------- |
| Linux x86-64  | Supported      |
| Linux aarch64 | Supported      |
| macOS         | Supported      |
| Windows       | Use containers |
| WSL2          | Supported      |

## Requirements

Before you start, make sure you have:

- [Python](https://realpython.com/installing-python/) 3.11 or higher
- A POSIX-compatible shell, such as bash or zsh
- [curl](https://curl.se/) (used by the installer to download dependencies)

## Installation Steps

### 1. Clone and Navigate into the Prometheus Repository

```sh
git clone https://github.com/Harvard-Neutrino/prometheus.git && cd prometheus
```

### 2. Run the Installer

```sh
bash install.sh
```

This sets up a water-based simulation environment, which is the recommended starting point. If you also need ice-based simulations, run:

```sh
bash install.sh --with-ppc
```

## What Gets Installed

The installer takes care of everything automatically:

- Creates an isolated environment in the `.prometheus_env` directory.
- Installs all Python dependencies.
- Builds the required scientific libraries:
    - **[PROPOSAL](https://github.com/tudo-astroparticlephysics/PROPOSAL)** — lepton propagation
    - **[LeptonInjector](https://github.com/icecube/LeptonInjector)** — neutrino interaction generation
- Optionally builds **[ppc](https://github.com/icecube/ppc)** — photon propagation for ice simulations (when run with `--with-ppc`).

## Activate the Environment

Once the installation is done, activate the environment:

```sh
source scripts/activate.sh .prometheus_env
```

!!! note
    You'll need to activate the environment in every new terminal session. To avoid this, add the command above to your shell profile (`~/.bashrc` or `~/.zshrc`).

## Verify the Installation

With the environment active, run an example:

```sh
python examples/01_basic_water.py
```

If everything went well, you should see simulation output without errors. If you run into problems, check that the environment is active first.

## Troubleshooting

- **Build failures (PROPOSAL / LeptonInjector)**: [Install with Containers](containers.md) instead.
- **Missing optional Python packages**: Some examples require `uproot` (for reading ROOT files) or `pandas` (for data manipulation). If you see an `ImportError` for either, install them with `pip install uproot` or `pip install pandas`.

## Getting Help

If you're having issues installing Prometheus, feel free to create [a discussion on GitHub](https://github.com/Harvard-Neutrino/prometheus/discussions) and we'll get back to you as soon as we can.
