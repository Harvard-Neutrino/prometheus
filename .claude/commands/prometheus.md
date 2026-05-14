# Prometheus — project skill

Use this skill whenever working in the Prometheus repository. It covers how to run
code, the project structure, and the conventions you must follow.

---

## Environment

**Python interpreter** (use paths relative to repo root — never rely on `python` or
`python3` being on PATH):

```
.prometheus_env/bin/python
```

**Pytest binary:**

```
.prometheus_env/bin/pytest
```

**pip:**

```
.prometheus_env/bin/pip
```

**Activate for interactive terminal sessions** (not needed when using full paths above):

```sh
source scripts/activate.sh .prometheus_env
```

The environment is a micromamba-managed conda env at `.prometheus_env/` (Python 3.12).

**Working directory:** Always run commands from the repo root. `conftest.py` enforces
this for pytest, and resource paths (geo files, model pickles) are relative to the
repo root.

---

## Running tests

```sh
# Standard suite (fast tests only)
.prometheus_env/bin/pytest tests/ -x

# Single file, verbose, with stdout
.prometheus_env/bin/pytest tests/test_compare_norm_flow.py -s -v

# Timing benchmarks (prints table to stdout)
.prometheus_env/bin/pytest tests/test_bench_water.py --timing -s

# Full end-to-end (slow, ~minutes)
.prometheus_env/bin/pytest tests/test_e2e.py --run-slow

# Linting
.prometheus_env/bin/ruff check prometheus/ tests/
```

Pytest markers (defined in `pyproject.toml`):
- `slow` — end-to-end simulation tests; deselected unless `--run-slow` is passed

---

## Project structure

```
prometheus/                        # main Python package
├── prometheus.py                  # Prometheus class; .sim() is the top-level entry point
├── config.py / config_types.py    # dataclass-based config; typed config tree
├── detector/                      # Detector, Medium enum, module definitions
├── injection/                     # Injection pipeline (LeptonInjector and GENIE modes)
│   ├── __init__.py                # INJECTORS registry (InjectorPlugin per injector)
│   ├── genie_injector.py          # GENIE ROOT file validator (runner for GENIE mode)
│   ├── genie_parser.py            # uproot-based gRooTracker reader → DataFrame
│   ├── lepton_injector_utils.py   # LeptonInjector runner
│   ├── registered_injectors.py    # RegisteredInjectors enum
│   └── injection/                 # Injection constructors
│       ├── genie_injection.py     # GENIEInjection, injection_from_genie_output()
│       └── LI_injection.py        # LIInjection, injection_from_LI_output()
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
├── resources/genie_example.root   # small gRooTracker ROOT file for GENIE tests
├── test_compare_norm_flow.py      # correctness + perf: reference vs fast norm flow
├── test_bench_water.py            # wall-time benchmark (--timing flag)
├── test_dataclasses.py            # unit tests for config dataclasses and enums
├── test_e2e.py                    # full pipeline end-to-end (--run-slow flag)
└── test_nflow_model.py            # unit tests for the normalizing flow model

examples/
├── 01_basic_water.py              # minimal single-event water simulation
├── 05_genie_injection.py          # Prometheus run driven by a GENIE ROOT file
├── water_config.toml              # example TOML config for water sims
└── legacy/                        # old scripts; excluded from ruff, not maintained
```

---

## Simulation pipeline

### LeptonInjector → Olympus (water)

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

### GENIE ROOT file injection

```
Prometheus.sim()
└─ inject()            ← reads gRooTracker ROOT file via uproot (no LeptonInjector)
   └─ injection_from_genie_output()
      ├─ genie_loader() → DataFrame of events
      ├─ vertex placement: "fixed" (user position / detector centre)
      │                 or "random" (uniform inside detector bounding cylinder)
      └─ π⁰ (PDG 111) decayed instantaneously to γγ at injection time
└─ propagate()         ← same photon propagation pipeline as above
```

**π⁰ note:** GENIE events may contain π⁰ final states. These are decayed to γγ at
injection time (instantaneous approximation; cτ ≈ 25 nm is negligible). A
`logger.warning` is emitted per π⁰ so users are aware.

**Output format:** `ak.Array` of shape `(n_modules,)` with ragged inner lists of float32
arrival times in nanoseconds. Modules with no hits have empty lists.

---

## Injector registry

New injectors are registered in `prometheus/injection/__init__.py`:

```python
INJECTORS: dict[RegisteredInjectors, InjectorPlugin] = {
    RegisteredInjectors.LEPTONINJECTOR: InjectorPlugin(runner=..., constructor=...),
    RegisteredInjectors.GENIE:          InjectorPlugin(runner=..., constructor=...),
}
```

`InjectorPlugin.runner` validates/runs injection before the simulation loop.
`InjectorPlugin.constructor` reads the injection data and returns an `Injection` object.

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
| `uproot` | Reads GENIE gRooTracker ROOT files |
| `LeptonInjector` | Neutrino event injection (C++ extension, loaded via injection/) |
| `hyperion` | JAX normalizing-flow model definitions (internal, at `hyperion/`) |
