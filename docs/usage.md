# Usage

## Running simulations

The [examples directory](https://github.com/Harvard-Neutrino/prometheus/tree/main/examples) of the Prometheus GitHub repository contains small, runnable scripts that demonstrate typical Prometheus workflows.

Recommended quick-start examples:

- `examples/01_basic_water.py` — Minimal water-case example to validate an install.
- `examples/02_basic_ice.py` — Minimal ice-case example (uses ppc) to validate an install with ppc.

To execute the examples, you need to activate the repository-local micromamba environment.

Run:

```sh
source scripts/activate.sh .prometheus_env
```

Then run an example script:

```sh
python examples/<SCRIPT_NAME>
```

Replace `<SCRIPT_NAME>` with the name of your script: `01_basic_water.py` or `02_basic_ice.py` for a quick-start option.

## Getting help

If something is not working as expected, or you have a question about using this software, feel free to create [a discussion on GitHub](https://github.com/Harvard-Neutrino/prometheus/discussions) and we will address it as soon as we can.

If you found a bug or want to suggest a change, feel free to [open an issue on GitHub](https://github.com/Harvard-Neutrino/prometheus/issues/new/choose) or make a contribution.

More information on contributing to Prometheus is available in our [contribution guidelines](https://github.com/Harvard-Neutrino/prometheus/blob/main/CONTRIBUTING.md).
