# Prometheus — project skill

Use this skill whenever working in the Prometheus repository. It covers how to run
code, the project structure, and the conventions you must follow.

---

## Environment

**Python interpreter** (use this path directly in every Bash tool call — never rely on
`python` or `python3` being on PATH):

```
/home/smeighenberger/projects/prometheus/.prometheus_env/bin/python
```

**Pytest binary:**

```
/home/smeighenberger/projects/prometheus/.prometheus_env/bin/pytest
```

**pip:**

```
/home/smeighenberger/projects/prometheus/.prometheus_env/bin/pip
```

**Activate for interactive terminal sessions** (not needed when using full paths above):

```sh
source scripts/activate.sh .prometheus_env
```

The environment is a micromamba-managed conda env at `.prometheus_env/` (Python 3.12).

**Working directory:** Always run commands from the repo root
(`/home/smeighenberger/projects/prometheus`). `conftest.py` enforces this for pytest,
and resource paths (geo files, model pickles) are relative to the repo root.

---

## Running tests

```sh
# Standard suite (fast tests only)
/home/smeighenberger/projects/prometheus/.prometheus_env/bin/pytest tests/ -x

# Single file, verbose, with stdout
/home/smeighenberger/projects/prometheus/.prometheus_env/bin/pytest tests/test_compare_norm_flow.py -s -v

# Timing benchmarks (prints table to stdout)
/home/smeighenberger/projects/prometheus/.prometheus_env/bin/pytest tests/test_bench_water.py --timing -s

# Full end-to-end (slow, ~minutes)
/home/smeighenberger/projects/prometheus/.prometheus_env/bin/pytest tests/test_e2e.py --run-slow

# Linting
/home/smeighenberger/projects/prometheus/.prometheus_env/bin/ruff check prometheus/ tests/
```

Pytest markers (defined in `pyproject.toml`):
- `slow` — end-to-end simulation tests; deselected unless `--run-slow` is passed
- `timing` — timing benchmarks; deselected unless `--timing` is passed

---

## Project structure

```
prometheus/                        # main Python package
├── prometheus.py                  # Prometheus class; .sim() is the top-level entry point
├── config.py / config_types.py    # dataclass-based config; OlympusSimConfig, etc.
├── detector/                      # Detector, Medium enum, module definitions
├── injection/                     # LeptonInjector wrapper
├── lepton_propagation/            # PROPOSAL wrapper
├── particle/                      # Particle representations
└── photon_propagation/
    ├── olympus_photon_propagator.py        # OlympusPhotonPropagator; wires the pipeline
    ├── photon_propagator.py               # abstract base class
    └── olympus/event_generation/
        ├── event_generation.py            # generate_cascade(), generate_realistic_track()
        ├── lightyield.py                  # fennel_total_light_yield(), make_realistic_cascade_source()
        ├── detector.py                    # sample_cylinder_volume(), generate_noise()
        ├── constants.py                   # c_vac, etc.
        └── photon_propagation/
            ├── norm_flow_photons.py       # REFERENCE implementation of generate_norm_flow_photons
            ├── norm_flow_photons_fast.py  # FAST variant (vectorised batch sampling; smb-water-speed branch)
            └── utils.py                   # sources_to_model_input (jit+vmap)

hyperion/                          # internal JAX normalizing-flow models
├── models/photon_arrival_time_nflow/net.py  # make_shape_conditioner_fn, traf_dist_builder, etc.
├── medium.py                      # medium_collections["pone"] → (ref_ix_f, sca_a_f, sca_l_f)
└── constants.py

resources/
├── geofiles/demo_water.geo        # demo P-ONE-style detector geometry
└── olympus_resources/
    ├── photon_arrival_time_nflow_params.pickle   # flow model weights
    └── photon_arrival_time_counts_params.pickle  # photon count network weights

tests/
├── conftest.py                    # chdir to repo root, --run-slow / --timing flags
├── test_compare_norm_flow.py      # correctness + perf: reference vs fast norm flow
├── test_bench_water.py            # wall-time benchmark (--timing flag)
├── test_e2e.py                    # full pipeline end-to-end (--run-slow flag)
└── test_nflow_model.py            # unit tests for the normalizing flow model

examples/
├── 01_basic_water.py              # minimal single-event water simulation
├── water_config.toml              # example TOML config for water sims
└── legacy/                        # old scripts; excluded from ruff, not maintained
```

---

## Simulation pipeline (water)

```
Prometheus.sim()
└─ inject()            ← LeptonInjector generates neutrino events
└─ propagate()
   └─ OlympusPhotonPropagator.propagate(particle, rng_key)
      ├─ [tracks]  generate_realistic_track()
      │            └─ PROPOSAL → stochastic + continuous energy losses
      │            └─ make_pointlike_cascade_source() / make_realistic_cascade_source()
      │            └─ generate_norm_flow_photons()
      └─ [cascades] generate_cascade()
                   └─ make_realistic_cascade_source()
                      └─ fennel_total_light_yield()     (Cherenkov photon count)
                      └─ fennel_frac_long_light_yield() (longitudinal profile)
                   └─ generate_norm_flow_photons()
                      ├─ sources_to_model_input()       (JAX jit+vmap: distances, angles)
                      ├─ distance mask (<300 m)
                      ├─ counts_net  → photon survival fraction
                      ├─ Poisson sample → per-(source,module) photon counts
                      ├─ flow conditioner → traf_params per pair
                      └─ sample arrival times per pair → ak.Array (ragged, per module)
```

**Output format:** `ak.Array` of shape `(n_modules,)` with ragged inner lists of float32
arrival times in nanoseconds. Modules with no hits have empty lists.

---

## Key constants and conventions

| Thing | Value / location |
|---|---|
| Distance cutoff | 300 m (normalizing flow not trained beyond this) |
| Source bucket base | power-of-2 (`_next_bucket`) |
| Wavelength default | 700 nm → c_medium ≈ 0.2174 m/ns |
| Model files | `resources/olympus_resources/*.pickle` |
| Medium key | `"pone"` (Cascadia Basin water; from `hyperion.medium.medium_collections`) |
| Cascade resolution | 0.2 m longitudinal step |
| Noise window | 5000 ns |
| BATCH_CAP (fast) | 1024 photons; pairs above this fall back to sequential sampling |

---

## Current branch: `smb-water-speed`

Focus: memory and calculation-time optimisation of `generate_norm_flow_photons`.

Completed changes:
- Replaced per-pair `jnp.repeat` approach with per-pair sequential sampling loop
  (avoids O(total_photons × num_flow_params) memory)
- Power-of-2 bucketing for JAX kernel reuse across events
- Removed broken `splitter` module-splitting logic from `generate_cascade` and
  `generate_realistic_track` (was using `%` instead of `//`, dead code for realistic detectors)

In progress:
- `norm_flow_photons_fast.py` — vectorised batch sampling via `jax.vmap` over all pairs
  (reduces n_masked sequential JAX dispatches to O(log max_photons) kernel calls)
- `tests/test_compare_norm_flow.py` — correctness + performance comparison harness
  (photon counts must be bit-for-bit identical; arrival times checked by KS test)

Pending improvements (see session context):
- Replace O(n_sources × n_modules) distance matrix in `generate_realistic_track` with KDTree
- Pre-split all PRNG keys before sampling loop (already in fast variant)
- CPU-side scatter for `n_ph_per_mod` (already in fast variant)
- Merged early-exit: single device sync for both the zero-count check and the loop

---

## Docstring style

NumPy format. Summary line must not start with "make". Use "create" or "build" for
factory functions. See memory file `feedback_docstring_style.md` for full rules.

---

## External dependencies to know

| Dep | Role |
|---|---|
| `jax` / `jaxlib` | JIT, vmap, PRNG, array ops throughout photon propagation |
| `proposal` ≥ 7.6.2 | Muon/tau energy loss (`pp.particle`, `propagator.propagate`) |
| `fennel_seed` | Cherenkov light yield integrals (`Fennel`, `auto_yields`) |
| `awkward` ≥ 2.6 | Ragged per-module photon time arrays (`ak.Array`) |
| `LeptonInjector` | Neutrino event injection (C++ extension, loaded via injection/) |
| `hyperion` | JAX normalizing-flow model definitions (internal, at `hyperion/`) |
