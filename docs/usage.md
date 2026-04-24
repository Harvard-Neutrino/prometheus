# Usage

## Running Simulations

The [examples directory](https://github.com/Harvard-Neutrino/prometheus/tree/main/examples) contains small, runnable scripts that demonstrate typical Prometheus workflows.

Recommended quick-start examples:

- `examples/01_basic_water.py` — Minimal water-case example to validate an install.
- `examples/02_basic_ice.py` — Minimal ice-case example (uses PPC) to validate an install with PPC.

Run examples using the repository-local micromamba environment. For a typical run:

Activate

```sh
# Activate the repo-local micromamba environment
source scripts/activate.sh .prometheus_env
```

Run a water simulation

```sh
# Run the water example
python examples/01_basic_water.py
```

Run an ice simulation

```sh
# Run the ice example
python examples/02_basic_ice.py
```

You can also combine activation and execution on one line:

```sh
source scripts/activate.sh .prometheus_env && python examples/01_basic_water.py
```

## Getting Help

If something is not working as expected, or you have a question about using this software, feel free to create [a discussion on GitHub](https://github.com/Harvard-Neutrino/prometheus/discussions) and we will address it as soon as we can.

If you found a bug or want to suggest a change, feel free to [open an issue on GitHub](https://github.com/Harvard-Neutrino/prometheus/issues/new/choose) or make a contribution.

More information on contributing to Prometheus is available in our [contribution guidelines](https://github.com/Harvard-Neutrino/prometheus/blob/main/CONTRIBUTING.md).
