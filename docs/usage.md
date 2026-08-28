# Usage

## Running Simulations

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

## Memory on the GPU

Prometheus never batches events together. The photon propagator runs one
particle at a time, so what sets peak device memory is the single brightest
particle in the run, not the number of events. A high-energy cascade close to
the detector can produce millions of detected photons, and the arrival-time
flow used to be evaluated for all of them in one call.

The relevant settings live under `config.photon_propagator.olympus.simulation`:

| Setting | Default | Effect |
| --- | --- | --- |
| `photon_chunk` | `262144` | Most photons handed to the arrival-time sampler per call. This bounds the sampler's working set to a fixed size however bright the particle. Lower it on a small GPU. |
| `module_chunk` | `128` | Modules evaluated per model-input call. Keeps that temporary, and the number of compiled kernels, independent of detector size. |
| `max_distance` | `300.0` | Drops source-module pairs further apart than this before propagation. Changing it changes physics, not just memory. |

Under `config.run`:

| Setting | Default | Effect |
| --- | --- | --- |
| `jax_release_interval` | `0` (off) | Drop the compiled-executable cache every this many propagations. Pair it with `jax_compilation_cache_dir` so recompilation is a disk read. |

JAX itself reserves half of the GPU on start-up. Prometheus sets
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` only when the variable is not already in
the environment, so on a shared or small card you can override it:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python my_run.py
# or allocate on demand instead of reserving up front
XLA_PYTHON_CLIENT_PREALLOCATE=false python my_run.py
```

An out-of-memory error part way through a run is therefore a single bright
particle, not accumulated events. Lower `photon_chunk` first; if the error
comes from the model-input step instead, lower `module_chunk`.

## Getting Help

If something is not working as expected, or you have a question about using this software, feel free to create [a discussion on GitHub](https://github.com/Harvard-Neutrino/prometheus/discussions) and we will address it as soon as we can.

If you found a bug or want to suggest a change, feel free to [open an issue on GitHub](https://github.com/Harvard-Neutrino/prometheus/issues/new/choose) or make a contribution.

More information on contributing to Prometheus is available in our [contribution guidelines](https://github.com/Harvard-Neutrino/prometheus/blob/main/CONTRIBUTING.md).
