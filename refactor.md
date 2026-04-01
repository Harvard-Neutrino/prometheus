# Prometheus Refactoring Plan

This document describes the proposed refactoring of the `prometheus`, `hyperion`,
and `olympus` packages.

**Hard constraints**

- The `Prometheus(config).sim()` API is unchanged.
- Existing YAML config files continue to work.
- Physics output distributions are not changed (no edits to JAX photon MC, light-yield
  model, normalizing-flow model, or PROPOSAL propagation math).
- Examples `01_basic_water.py` and `02_basic_ice.py` produce equivalent output
  before and after every phase.

---

## Phase 0 — Safety net (prerequisite for all other phases)

Before touching any logic, establish a deterministic regression baseline:

1. Capture reference parquet checksums from both examples with a fixed RNG seed.
2. Add a minimal test suite under `tests/`:
   - Import smoke-test for every submodule.
   - One end-to-end run per example that asserts the checksum matches.
3. Every subsequent phase ends with re-running this suite.
4. Add a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push
   and pull request: install via `bash install.sh`, run the test suite, report
   pass/fail. This closes the loop so no future change can silently break the install.

**Also in this phase — close `.gitignore` gaps.** The following outputs and artefacts
are not currently excluded and could be accidentally committed:

```
PROPOSAL_tables/
examples/output/
*.lic
*.f2k
*.ppc
__pycache__/
*.pyc
.prometheus_env/
```

Add them to `.gitignore` before any other work begins.

---

## Phase 1 — Delete dead and broken code

The following files are either completely broken, superseded, or unreachable from
any live code path. Removing them reduces the total line count by over 25 % and
eliminates several misleading abstractions.

| File | Reason for removal |
|---|---|
| `olympus/event_generation/photon_propagation/legacy_photon_propagation.py` | Entirely commented-out / broken dead code (277 lines) |
| `prometheus/lepton_propagation/old_proposal_lepton_propagator.py` | PROPOSAL v6-only; only v7.6.2 is supported and installed |
| `prometheus/lepton_propagation/registered_propagators.py` | Single-propagator world after v6 removal; the enum served no purpose |
| `prometheus/utils/hebe_ui.py` | References config keys that no longer exist; imports `colorama` (not in dependencies); fully non-functional |
| `prometheus/__main__.py` | Broken import (`from utils.hebe_ui import run_ui`); `python -m prometheus` has never worked |
| `hyperion/models/photon_arrival_time_torch/` | Skeleton stubs with no implementation; never imported by the pipeline |

Add a `resources/README.md` explaining the purpose of each subdirectory
(`LeptonInjector/`, `fennel/`, `PPC_executables/`, cross-sections, etc.), the
vendoring policy, and how to update a vendored dependency. This is a one-hour
documentation task that pays dividends every time a new contributor asks "why is
this source code checked in?".

Removing `old_proposal_lepton_propagator.py` also allows the PROPOSAL version-detection
import that currently runs at **module load time** in
`prometheus/lepton_propagation/__init__.py` to be made lazy (moved inside the
propagator constructor). This stops every `import prometheus` from triggering a full
PROPOSAL import.

---

## Phase 2 — Fix outright non-physics bugs

These are correctness bugs with no effect on physics output.

**2a. Abstract base classes not enforced**
Both `LeptonPropagator` and `PhotonPropagator` declare `@abstractmethod` methods but
inherit from `object` instead of `abc.ABC`, so subclasses are never required to
implement the interface. Fix: add `abc.ABC` as base class.

**2b. `InvalidRNGError.__init__` in `detector_factory.py`**
`super.__init__(message)` calls the unbound descriptor rather than the constructor.
Fix: `super().__init__(message)`.

**2c. `ConfigClass.from_yaml` / `from_dict` shallow merge**
A user YAML containing `run: {nevents: 5}` silently wipes all other `run.*` defaults
because `dict.update` is not recursive. Both methods carry a `# TODO: Update this`
comment acknowledging this bug. Fix: a ~10-line `deep_merge` helper.

**2d. `PPC_CUDA` config key typo**
The key `"ppc_tmpdir:"` in `config.py` (note trailing colon) causes a `KeyError` at
runtime for any PPC_CUDA run. Fix: remove the spurious colon.

**2e. Duplicate unit constants**
`MeV_to_GeV` and `GeV_to_MeV` are each defined twice in `utils/units.py`.
Fix: remove the duplicates.

**2f. Hardcoded LI install path**
The default `install location` in `config.py` is the absolute path
`/opt/LI/install/lib/python3.9/site-packages`, which hardcodes both a machine-specific
prefix and a Python 3.9 version string. Fix: derive at runtime from `sys.prefix`.

---

## Phase 3 — Consolidate duplicated code

**3a. `olympus/event_generation/detector.py` vs `prometheus/detector/`**
The five geometry builder functions (`make_line`, `make_grid`, `make_hex_grid`,
`make_triang`, `make_rhombus`) and the `Module`/`Detector` class structures exist in
both packages with diverging signatures. The prometheus versions are current; the
olympus versions are the originals. Fix: reduce `olympus/event_generation/detector.py`
to re-exports from `prometheus/detector/`, keeping the names intact for backward
compatibility.

**3b. `INTERACTION_DICT` in `config_mims.py` vs `INTERACTION_CONVERTER` in `LI_injection.py`**
Two overlapping mappings of particle name strings to interaction enum values.
Consolidate into one canonical dict in `prometheus/injection/interactions.py`.

**3c. Geometry helpers overlap**
`olympus/event_generation/utils.py` and `prometheus/utils/geo_utils.py` contain
overlapping track/cylinder geometry helpers. Move any unique olympus helpers into
`prometheus/utils/geo_utils.py` and reduce the olympus file to re-exports.

---

## Phase 4 — Config system cleanup

**4a. Remove runtime state from the config dict**
`prometheus.py` currently injects
`config["photon propagator"]["olympus"]["runtime"] = {"random state": ..., "detector": ...}`
at simulation time, making the config non-serializable and non-reentrant. The random
state and detector reference should be passed as explicit arguments to
`propagator.propagate()` instead.

**4b. Config mutation in `config_mims.py`**
`del config["simulation"]` when `inject=False` irreversibly mutates the caller's dict.
Fix: operate on a shallow copy.

**4c. Implement `check_consistency`**
Currently a `pass` stub. Implement basic validation: required keys present, numeric
ranges plausible, file paths that must exist actually exist. This replaces cryptic
downstream `KeyError`s with clear messages.

---

## Phase 4b — Typed configuration objects

The current config system is a nested `dict` built in `config.py` and passed everywhere
by reference. The problems that arise from this are well-known: unknown keys are silently
ignored, typos are only caught at runtime several layers deep, there is no IDE
autocomplete, and the shape of the config object can only be understood by reading the
full default dict.

Replace the nested dict with `dataclasses` (or Pydantic v2 models — either works;
`dataclasses` has no extra dependency).

```python
@dataclass
class RunConfig:
    nevents: int = 100
    random_state_seed: int = 1337
    verbosity: str = "WARNING"

@dataclass
class SimulationConfig:
    final_state_1: str = "MuMinus"
    final_state_2: str = "Hadrons"
    minimal_energy: float = 1e2   # GeV
    maximal_energy: float = 1e6   # GeV
    # ...

@dataclass
class PrometheusConfig:
    run: RunConfig = field(default_factory=RunConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    lepton_propagator: LeptonPropagatorConfig = field(default_factory=LeptonPropagatorConfig)
    photon_propagator: PhotonPropagatorConfig = field(default_factory=PhotonPropagatorConfig)
```

A `PrometheusConfig.from_yaml(path)` class method performs a deep merge of user keys
over the defaults (fixing Phase 2c at the same time), and unknown keys raise
`TypeError` at construction time instead of silently doing nothing.

This phase is the highest-value single change for long-term maintainability. It should
be done after Phase 4 (config cleanup) and before Phase 5 (folding olympus in), so
the new layout reflects the cleaned-up config structure.

---

## Phase 5 — Fold olympus into prometheus

Olympus is only ever called through `OlympusPhotonPropagator`. It has no separate
user-facing API and no reason to be an independent top-level package. Folding it in
eliminates the cross-package confusion and the duplicate detector code described in
Phase 3a.

Proposed new layout:

```
prometheus/
    photon_propagation/
        olympus/                  ← renamed from top-level olympus/
            event_generation/
            lightyield.py
            ...
        olympus_photon_propagator.py
        ppc_photon_propagator.py
        ...
```

The `olympus` top-level name becomes an import shim (`olympus/__init__.py` re-exports
from `prometheus.photon_propagation.olympus`) for any external code that imports it
directly — though audit of the examples shows none do.

`hyperion` stays as a standalone package since it is a self-contained JAX physics
library (MC propagation, optical property models, ML models) that could independently
be useful and has no imports from prometheus or olympus.

### Propagator plugin registry

Currently adding a new photon propagator requires edits to four separate files
(`registered_photon_propagators.py`, `__init__.py`, `prometheus.py`, and the config
default dict). Replace this with a class-decorator registry:

```python
# prometheus/photon_propagation/registry.py
_PROPAGATOR_REGISTRY: dict[str, type] = {}

def register_propagator(name: str):
    def decorator(cls):
        _PROPAGATOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_propagator(name: str):
    try:
        return _PROPAGATOR_REGISTRY[name.lower()]
    except KeyError:
        raise UnknownPhotonPropagatorError(name)
```

Each propagator class gains `@register_propagator("olympus")` (or `"ppc"`, `"ppc_cuda"`).
`prometheus.py` calls `get_propagator(config["photon propagator"]["name"])` and
instantiates it directly. Adding a new propagator then requires touching only one file.

The same pattern applies to lepton propagators (`@register_lepton_propagator`) and
injectors (`@register_injector`), making Phase 13 (flexible injection) easier to
implement.

---

## Phase 5b — Particle test isolation (`clone()`)

`Particle` objects are mutated throughout the pipeline: `.hits`, `.children` and
`.losses` are set in-place on the same object that is passed through lepton
propagation, photon propagation, and output serialisation. This makes it impossible
to write an isolated unit test for any single stage without running the full pipeline
first to produce a particle in the correct state.

The full API-breaking fix (immutable particles with builder methods) is deferred
because it touches every stage. The intermediate fix that unblocks testing now is a
`clone()` method:

```python
class Particle:
    def clone(self) -> "Particle":
        """Return a deep copy with independent hit/child/loss lists."""
        import copy
        return copy.deepcopy(self)
```

With `clone()` available, test fixtures can produce a particle in the desired state
once and each test case clones it before mutating, giving full isolation without
changing any production code paths.

The longer-term work (if it ever becomes necessary) would be immutable `ParticleState`
snapshots — design that decision when the test suite is mature enough to validate it.

---

## Phase 6 — Water medium model

### How new water models are added (current infrastructure)

`hyperion/medium.py` already provides all the factory functions needed. Adding a new
water medium requires only three steps:

1. **Define the physical parameters** — salinity (ppt), temperature (°C), pressure (bar),
   and volumetric particle concentrations for the scattering length model.

2. **Call the factory functions**:
   ```python
   my_ref_index  = make_ref_index_func(salinity, temperature, pressure)
   my_sca_len    = make_wl_dep_sca_len_func(vol_conc_small, vol_conc_large)
   ```
   A scattering angle function is chosen from the existing set
   (`mixed_hg_rayleigh_antares`, `mixed_hg_liu_icecube`) or a new one is composed
   with `make_mixed_scattering_func`.

3. **Register the medium**:
   ```python
   medium_collections["my_medium"] = (my_ref_index, scattering_angle_fn, my_sca_len)
   ```
   and add the corresponding key to the `Medium` enum and the lookup in
   `olympus_photon_propagator.py`.

No changes to the propagation code are required. The medium functions are pure JAX
callables, so they are JIT-compiled automatically.

Pre-built refractive index functions for two environments are already in the file
and unused in `medium_collections`:

| Name | Parameters |
|---|---|
| `antares_ref_index_func` | pressure 215 bar, temp 13.1 °C, salinity 38.44 ppt (Mediterranean) |
| `cascadia_ref_index_func` | pressure 269 bar, temp 1.8 °C, salinity 34.82 ppt (NE Pacific / P-ONE) |

Currently only `"pone"` (Cascadia Basin) is registered.

### P-ONE medium and other water detectors

The current code silently overwrites `medium_collections[detector.medium]` with the
P-ONE entry for any water-based detector regardless of what medium was requested. This
is acknowledged in the code with `# TODO This is reeeeeeeally bad I think`.

The fix (Phase 4 / Phase 5) replaces the silent overwrite with an explicit mapping. Where
a detector medium has no dedicated registered model, instead of crashing a warning is
printed:

```
Warning: No dedicated optical model is registered for medium 'ARCA'.
         Falling back to the Cascadia Basin (P-ONE) water model.
         Simulation results may not accurately reflect ARCA optical properties.
```

This preserves the current behaviour (use P-ONE as default) while making it visible.
As new registered media are added (Mediterranean for ARCA/ORCA, Baikal for GVD) the
warning disappears automatically for those detectors.

---

## Phase 7 — Eliminate distrax and dm-haiku

These two libraries are currently blocking numpy ≥ 2, newer JAX, and ultimately
Python 3.13 support. Both are used **only** in
`hyperion/models/photon_arrival_time_nflow/net.py`. Nothing else in the pipeline
touches them.

### Exact usage inventory

**From `distrax`** (all in `net.py`):

| Class | Role |
|---|---|
| `distrax.RationalQuadraticSpline(params, range_min, range_max)` | Per-layer normalizing flow bijector |
| `distrax.Gamma(1.5, 0.1)` | Base distribution for the flow |
| `distrax.ScalarAffine(shift, scale)` | Final affine bijector |
| `distrax.Inverse(bijector)` | Invert a bijector |
| `distrax.Chain(bijectors)` | Compose a list of bijectors |
| `distrax.Transformed(base, flow)` | Combine base distribution + bijector chain |

**From `dm-haiku`** (all in `net.py`):

| Symbol | Role |
|---|---|
| `hk.Linear(n_out, w_init, b_init)` | Dense output layer |
| `hk.Flatten(preserve_dims=1)` | Reshape layer |
| `hk.nets.MLP(hidden_sizes, activate_final)` | Hidden MLP stack |
| `hk.Sequential([...])` | Layer container |
| `hk.without_apply_rng` / `hk.transform` | Turn a function into (init, apply) pair |
| `hk.PRNGSequence(seed)` | Random key sequence (training only) |

The inference path (loading pickled weights and running `apply_fn` / `sample_model`)
uses only `hk.without_apply_rng`, `hk.transform`, and `hk.Linear` /
`hk.nets.MLP` through the saved parameter tree. The training functions (`train_shape_model`,
`train_counts_model`) additionally use `hk.PRNGSequence`.

### Replacement strategy

**Option A — Vendor the used primitives (lowest risk)**

Copy the ~6 distrax classes and ~5 haiku wrappers that are actually used into
`prometheus/_vendor/mini_distrax.py` and `prometheus/_vendor/mini_haiku.py`.
The implementations are pure JAX math — they are stable and will not change. The
pickled model weights are arrays keyed by string paths in a nested dict; they are
independent of the class names used to load them. Inference code continues working
with zero physics changes.

Effort: ~350 lines of vendor code. Risk: very low.

**Option B — Migrate to TFP-on-JAX + Flax (cleanest long-term)**

`tensorflow_probability` is already installed as a transitive dependency. Its JAX
substrate (`tfp.substrates.jax`) provides exact drop-in replacements:

| distrax | tfp.substrates.jax |
|---|---|
| `distrax.RationalQuadraticSpline` | `tfp.bijectors.RationalQuadraticSpline` |
| `distrax.Gamma` | `tfp.distributions.Gamma` |
| `distrax.Chain` | `tfp.bijectors.Chain` |
| `distrax.Transformed` | `tfp.distributions.TransformedDistribution` |
| `distrax.Inverse` | `tfp.bijectors.Invert` |
| `distrax.ScalarAffine` | `tfp.bijectors.Scale` + `tfp.bijectors.Shift` |

For haiku, migrate the conditioner MLP to `flax.linen` (Flax is already installed
in the environment):

| haiku | flax.linen |
|---|---|
| `hk.transform` / `hk.without_apply_rng` | `flax.linen.Module` with `@nn.compact` |
| `hk.Linear` | `flax.linen.Dense` |
| `hk.nets.MLP` | `flax.linen.Sequential` + `flax.linen.Dense` |
| `hk.Sequential` | `flax.linen.Sequential` |
| `hk.PRNGSequence` | `jax.random.split` directly |

**Critical constraint**: the pickled model files store a Haiku parameter tree whose
structure is `{"linear": {"w": ..., "b": ...}, "mlp/~/linear_0": {"w": ..., ...}}`.
Flax uses a different key scheme (`{"params": {"Dense_0": {"kernel": ..., "bias": ...}}}`).
A one-time weight-migration script must be written and the pickled files re-saved
before switching to Flax for inference.

Effort: ~150 lines of new model code + one-time weight migration script. Risk: moderate
(requires thorough NF output validation after migration).

**Recommendation**: do Option A first to immediately unblock the numpy/JAX/Python
version upgrades, then do Option B as a follow-up once the test suite (Phase 0) has
sufficient coverage to validate the NF output.

---

## Phase 8 — Dependency modernization

With distrax and haiku removed (Phase 7), the following pins can be relaxed:

| Dependency | Current pin | Proposed | Notes |
|---|---|---|---|
| `numpy` | `==1.26.4` | `>=1.26,<3` | Hard pin blocks all modern packages |
| `pyarrow` | `==15.0.2` | `>=15` | Hard pin 15 months old; no breaking changes |
| `jax` / `jaxlib` | `==0.4.35` | `>=0.4.35` | Lower bound only; upper bound removed once distrax/haiku gone |
| `proposal` | `==7.6.2` | `>=7.6.2,<8` | Allow patch releases |
| `distrax` | `^0.1.5` | *removed* | Replaced in Phase 7 |
| `dm-haiku` | `^0.0.11` | *removed* | Replaced in Phase 7 |
| `uproot` / `pandas` | missing | optional extra `prometheus[genie]` | Used by `genie_parser.py` but undeclared |

---

## Phase 9 — Minor cleanup

Lower risk, high readability gain. Can be done incrementally alongside other phases.

- **Logging infrastructure.** Replace all `print()` calls with the standard
  `logging` module. Add a `run: verbosity: WARNING` config key (default `WARNING`).
  `Prometheus.__init__` calls `logging.basicConfig(level=config["run"]["verbosity"])`
  once. The ~20 scattered `print()` calls throughout `prometheus.py`, `config_mims.py`,
  and the photon propagators become `logger.debug(...)` or `logger.info(...)` as
  appropriate. Users needing diagnostic output pass `verbosity: DEBUG` in their YAML.
- Replace `print("I am melting.... AHHHHHH!!!!")` in `Prometheus.__del__` with
  `logger.debug("Prometheus instance finalised.")` (or remove entirely).
- Fix `recursively_get_final_property` to use a list accumulator and one
  `np.concatenate` at the end instead of `np.hstack` in a loop ($O(n^2)$).
- Fix `to_dict` in `Injection` to make a single pass instead of seven.
- Standardise on `pathlib.Path` throughout instead of mixed `os.path` / raw strings.
- Remove the duplicate `import uproot` lines in `genie_parser.py`.
- Fix typos `cyinder_radius` / `cyinder_height` in `lepton_injector_utils.py`.
- Replace remaining `# TODO`, `# FIXME`, and `# Sorry about this` comments with
  either an action or a `# NOTE:` explanation.

---

## Phase 10 — Document what hyperion is

Hyperion has no documentation beyond brief module docstrings and is currently
described nowhere in the project. Because it is also the part of the codebase most
likely to be used independently (e.g. by someone building a new water detector model
or training a new photon timing network), a clear standalone description belongs in
`docs/` and in the package itself.

### What hyperion is

`hyperion` is a self-contained JAX physics library for water-Cherenkov photon
simulation. It has no imports from `prometheus` or `olympus`. Its components are:

| Module | Role |
|---|---|
| `hyperion/propagate.py` | Full Monte Carlo photon transport. JIT-compiled JAX implementation of photon propagation through water, including absorption, Henyey-Greenstein/Rayleigh/Liu scattering, and sphere–photon intersection geometry. |
| `hyperion/medium.py` | Parametric optical property models. Factory functions for wavelength-dependent refractive index (salinity/temperature/pressure), scattering length (particle concentration), and scattering angle distributions. The `medium_collections` registry maps medium names to tuples of these three functions. |
| `hyperion/models/photon_arrival_time_nflow/net.py` | Normalizing flow photon timing model. An MLP-conditioned rational quadratic spline flow (trained offline on MC data) that predicts photon arrival time distributions given source–module geometry. Used at inference time by olympus. |
| `hyperion/models/photon_binned_amplitude/net.py` | Binned photon count model. Predicts expected photon yield per module in amplitude bins. |
| `hyperion/utils.py` | Cherenkov and rotation math. Time residual calculation, Cherenkov angular distribution, spherical-to-Cartesian frame rotation for JAX and numpy. |
| `hyperion/constants.py` | Physical constants. Speed of light, Cherenkov light yield (photons/GeV), PMT angular distribution coefficients. |
| `hyperion/pmt/pmt.py` | PMT response. Single-photoelectron template and waveform generation. |

### What to write

1. **`docs/prometheus/hyperion.md`** — prose description of the above, including a
   diagram of the data flow from a photon source through propagation to a hit, and
   a note on the JAX JIT compilation model (first call is slow; subsequent calls
   are compiled).
2. **`hyperion/__init__.py`** — currently empty. Add a module docstring summarising
   the package and listing its public API.
3. **A note in `docs/index.md`** explaining the three-package architecture:
   hyperion (JAX physics) → olympus (event generation, currently being folded
   in) → prometheus (orchestration and user API).

---

## Phase 11 — Water model example

Add a worked example showing how to define and use a new water optical model.
This is a documentation gap that is particularly important for groups wanting to
add support for their own detector (ARCA, ORCA, GVD, TRIDENT, etc.).

### What the example should cover

1. **Choose physical parameters** for the target water body. The required inputs
   to the existing factory functions are:

   | Parameter | Unit | Notes |
   |---|---|---|
   | `salinity` | parts per thousand (ppt) | Typical ocean: 35 ppt |
   | `temperature` | °C | Use the value at detector depth |
   | `pressure` | bar | 1 bar ≈ 10 m depth |
   | `vol_conc_small_part` | ppm | Volumetric concentration of small scattering particles |
   | `vol_conc_large_part` | ppm | Volumetric concentration of large scattering particles |

2. **Call the hyperion factory functions**:
   ```python
   from hyperion.medium import (
       make_ref_index_func,
       make_wl_dep_sca_len_func,
       mixed_hg_rayleigh_antares,   # or mixed_hg_liu_icecube, or compose your own
       medium_collections,
   )

   # Example: KM3NeT ARCA site (Mediterranean off Sicily)
   arca_ref_index = make_ref_index_func(
       salinity=38.5,
       temperature=13.5,
       pressure=350.0,          # ~3500 m depth
   )
   arca_sca_len = make_wl_dep_sca_len_func(
       vol_conc_small_part=0.0075,
       vol_conc_large_part=0.0075,
   )
   medium_collections["arca"] = (
       arca_ref_index,
       mixed_hg_rayleigh_antares,
       arca_sca_len,
   )
   ```

3. **Validate visually** — plot refractive index, scattering length, and speed of
   light in medium as a function of wavelength (300–700 nm) and compare against any
   published ARCA/site measurements. This validates the parameters before running
   a full simulation.

4. **Register the medium in the `Medium` enum** and map it in
   `olympus_photon_propagator.py` so it can be selected from a YAML config with
   `medium: arca`.

5. **Run a simulation** using the new medium and compare photon yields and timing
   against the P-ONE (Cascadia Basin) baseline.

### Deliverable

A Jupyter notebook at `examples/03_custom_water_model.ipynb` stepping through all
five points above, with inline plots of the optical properties and a brief simulation
run using the new medium. The notebook should be self-contained and runnable with
no external data beyond the installed packages.

---

## Phase 12 — GPU usage documentation

Prometheus does not currently document how a user can exploit GPU hardware.
This phase adds a `docs/prometheus/gpu.md` page covering the available options,
their requirements, and their trade-offs. No code changes are made.

### Water mode — NVIDIA CUDA (current production path)

The normalizing-flow photon model and all JAX-based physics in hyperion run on CPU
by default. A CUDA-capable NVIDIA GPU can be used by replacing the installed `jaxlib`
wheel with the CUDA variant:

```bash
pip install --upgrade "jaxlib[cuda12]"
```

JAX will then automatically dispatch all `@jax.jit`-compiled functions to the GPU.
No configuration changes or code changes are required. This is the most mature and
tested path.

### Ice mode — NVIDIA CUDA (PPC_CUDA)

The `PPC_CUDA` propagator invokes a CUDA-compiled PPC binary as a subprocess.
The binary is not bundled (it must be compiled with `nvcc` on a CUDA-capable machine).
The relevant config keys (`ppc_exe`, `ppc_tmpdir`, `ppctables`) point to the binary
and its working files. Examples of this setup are in `examples/example_ppc.py` and
`examples/fasrc_example_ppc.py`.

### Water mode — Intel GPU (experimental)

Intel Arc and Data Center GPUs can be targeted via
[`intel-extension-for-openxla`](https://github.com/intel/intel-extension-for-openxla),
a PJRT plugin that exposes Intel GPU hardware to JAX as `sycl` devices. Once
installed, `jax.devices()` lists `sycl` devices and `@jax.jit` code dispatches to
them without code changes.

Requirements and caveats to document:

| Item | Detail |
|---|---|
| Supported hardware | Officially: Intel Data Center GPU Max/Flex Series. Client GPUs (Arc) may work but are not officially verified. |
| JAX version | Requires JAX ≥ 0.4.38; prometheus currently pins 0.4.35 (relaxed in Phase 8b). |
| oneAPI toolkit | Requires Intel DPC++ compiler and oneMKL at runtime — a multi-GB install. |
| WSL2 | GPU passthrough must be enabled separately (requires `intel-compute-runtime` inside WSL2 and a recent Windows Intel Graphics driver). |
| No code changes needed | All hyperion `@jax.jit` functions run unmodified on `sycl` devices once the plugin is active. |

### Ice mode — OpenCL (PPC)

The upstream `icecube/ppc` repository ships three independent implementations of the
same photon transport physics:

| Directory | Backend | Notes |
|---|---|---|
| `gpu/` | CUDA or CPU | Vendored in `resources/PPC_executables/PPC/` |
| `ocl/` | OpenCL | Runs on any OpenCL 1.2+ device including Intel Arc, AMD, and older NVIDIA |
| `par/` | C++ `std::par` | CPU-only, parallel, no GPU driver required |

Because prometheus invokes PPC as a subprocess, switching to the OpenCL or `std::par`
binary requires only pointing `ppc_exe` in the config to the alternative binary — no
Python changes. The documentation should explain how to obtain and build each variant
from `icecube/ppc` and when each is appropriate.

### What to write

A single page `docs/prometheus/gpu.md` structured as:

1. **Overview** — which simulation modes benefit from a GPU and which don't.
2. **NVIDIA CUDA** — step-by-step for both water (jaxlib CUDA wheel) and ice
   (PPC_CUDA binary).
3. **Intel GPU** — the oneAPI/openxla setup for water mode; the OpenCL PPC build for
   ice mode. Clear note on WSL2 passthrough requirement and Arc support status.
4. **No GPU / CPU fallback** — confirm that both modes work on CPU (current default)
   and note the `par/` PPC variant as a parallel CPU alternative for ice.
5. **Troubleshooting** — common failures (`No visible XPU devices`, wrong jaxlib
   wheel, `/dev/dri/` absent).

---

## Phase 13 — Flexible injection and external data loading (GENIE)

### Motivation

Prometheus currently assumes injection is always performed by LeptonInjector (LI).
A user who has already run GENIE (or another neutrino event generator) and wants to
use those events as input must re-simulate injection from scratch with LI, losing
the GENIE interaction model entirely. The codebase already anticipates this use case:
`RegisteredInjectors.GENIE = 3` exists in the enum, `genie_parser.py` has all the
parsing logic (`genie_loader`, `genie_parser`, `final_parser`, `genie2prometheus`),
and the config has a `'GENIE'` section. What is missing is the wiring that connects
them.

### What currently exists

| File | Status |
|---|---|
| `prometheus/injection/registered_injectors.py` | `GENIE=3` in enum — exists |
| `prometheus/injection/genie_parser.py` | `genie_loader`, `genie2prometheus` — exists but untested |
| `prometheus/config.py` line 83 | `'GENIE': {...}` config section — exists but sparse |
| `prometheus/injection/__init__.py` | `INJECTOR_DICT` and `INJECTION_CONSTRUCTOR_DICT` — **GENIE missing** |
| `prometheus/prometheus.py` | `inject()` dispatches via the two dicts — **GENIE falls through** |

GENIE reads ROOT files in `gRooTracker` format via `uproot`. The `genie_loader`
function already opens the file and calls through to `genie2prometheus`, which
returns a list of events. The gap is that `genie2prometheus` does not yet construct
`InjectionEvent` objects and its return value is never passed to
`INJECTION_CONSTRUCTOR_DICT`.

### Design

The injection pipeline has two distinct responsibilities:

1. **Event generation** — either run LI or skip if external data is provided.
2. **File loading** — read a file (LI `.hdf5` or GENIE ROOT) and construct
   `Injection` objects.

Generalise this into a three-mode dispatch controlled by a single config key
`injection: name: <mode>`:

| Mode | Behaviour |
|---|---|
| `LeptonInjector` | Run LI → write `.hdf5` → parse `.hdf5` (current behaviour) |
| `GENIE` | Skip event generation → load ROOT file → parse via `genie2prometheus` |
| `Prometheus` | Skip event generation → load a previously-written Prometheus parquet injection output |

The `inject: true/false` flag already in the `LeptonInjector` config section
generalises to *all* injectors: `inject: false` means "load from file, do not run
the generator". For GENIE this is always `false` since Prometheus never invokes
GENIE itself.

### Implementation steps

**13a. Complete `genie2prometheus`**

The function currently returns a `pd.DataFrame`. It needs to convert rows into
`InjectionEvent` objects that the rest of the pipeline can consume. The existing
`LIInjectionEvent` dataclass can be reused or a lighter `GenieInjectionEvent` with
only the fields GENIE provides (vertex, initial-state PDG + momentum, final-state
PDGs + momenta) can be added.

Key mapping from GENIE `gRooTracker` format to `InjectionEvent` fields:

| GENIE field | InjectionEvent field |
|---|---|
| `EvtVtx[0:3]` | `vertex_x`, `vertex_y`, `vertex_z` |
| `StdHepPdg[0]` | `initial_state.pdg_encoding` |
| `StdHepP4[0][3]` | `initial_state.e` (total energy, GeV) |
| `StdHepPdg[status=="final"]` | `final_states[*].pdg_encoding` |
| `StdHepP4[status=="final"]` | `final_states[*].e` and `.direction` |

Note: GENIE ROOT files do not contain Bjorken-x/y or column depth. Set these to
`0.0` in the `LIInjectionEvent` or leave them out of a new `GenieInjectionEvent`.

**13b. Wire GENIE into `INJECTION_CONSTRUCTOR_DICT`**

```python
# prometheus/injection/__init__.py

from .genie_parser import genie_loader, genie2prometheus
from .injection import injection_from_genie_output  # new thin wrapper

INJECTION_CONSTRUCTOR_DICT = {
    RegisteredInjectors.LEPTONINJECTOR: injection_from_LI_output,
    RegisteredInjectors.GENIE:          injection_from_genie_output,
}
```

`injection_from_genie_output(filepath)` calls `genie_loader(filepath)` then wraps
the result in an `Injection` object. No changes to `prometheus.py` are needed
because it already dispatches via `INJECTION_CONSTRUCTOR_DICT`.

**13c. Update the GENIE config section**

```yaml
injection:
  name: GENIE
  GENIE:
    inject: false          # always false — Prometheus does not invoke GENIE
    paths:
      injection file: /path/to/gntp.root
```

The `inject: false` path in `prometheus.py::inject()` already skips the
`INJECTOR_DICT` call, so no generator is invoked, and proceeds directly to
`INJECTION_CONSTRUCTOR_DICT[GENIE](filepath)`.

**13d. Coordinate system alignment**

LI places the detector at the origin; GENIE places it at the GENIE geometry origin.
The `genie2prometheus` conversion must apply `detector.offset` to vertex coordinates,
the same way `injection_from_LI_output` does. Document the expected coordinate
convention in the config (`GENIE: paths: detector_offset: [x, y, z]`).

**13e. Declare optional dependencies**

`uproot` and `pandas` are used by `genie_parser.py` but not listed in
`pyproject.toml`. Add them as an optional extra (already noted in Phase 8):

```toml
[project.optional-dependencies]
genie = ["uproot>=5", "pandas>=2"]
```

Users who do not use GENIE do not need these installed.

**13f. Add example**

`examples/04_genie_loading.py` — a minimal script that:
1. Loads a small GENIE ROOT file (include a tiny synthetic one in `resources/test_data/`
   for CI).
2. Constructs a `Prometheus` instance with `injection: name: GENIE`.
3. Calls `sim()` and asserts non-zero photon hits.

This becomes the third regression test in Phase 0's CI workflow once the file is added.

### Extensibility note

The same pattern supports any future external generator (NuGen, NuFlux, CORSIKA, etc.):
add an entry to `RegisteredInjectors`, write a parser returning `InjectionEvent` objects,
add the entry to `INJECTION_CONSTRUCTOR_DICT`, and optionally use the
`@register_injector` decorator from Phase 5 to eliminate even that step. No changes
to `prometheus.py` or the core pipeline are ever needed.

---

## Implementation order and risk summary

| Phase | Physics risk | API risk | Effort |
|---|---|---|---|
| 0 — Safety net + CI + .gitignore | None | None | Small |
| 1 — Delete dead code + resources/README | Low | None | Small |
| 2 — Fix non-physics bugs | None | None | Small |
| 9 — Minor cleanup + logging | None | None | Small |
| 3 — Consolidate duplicates | Low | None | Medium |
| 4 — Config cleanup | None | None (same YAML) | Medium |
| 4b — Typed config objects | None | None (same YAML) | Medium |
| 5 — Fold olympus in + propagator registry | None | None (API unchanged) | Medium |
| 5b — Particle.clone() | None | None | Small |
| 13 — Flexible injection / GENIE | None | None | Medium |
| 6 — Water media | None | None | Small |
| 8a — Unpin numpy/pyarrow | Low (test carefully) | None | Small |
| 7a — Vendor distrax/haiku | Very low | None | Medium |
| 8b — Relax jax pin | Low | None | Small |
| 7b — Migrate to TFP/Flax | Moderate (NF output) | None | Large |
| 10 — Hyperion documentation | None | None | Small |
| 11 — Water model example | None | None | Small |
| 12 — GPU usage documentation | None | None | Small |

**Recommended sequencing**: 0 → 1 → 2 → 9 → 3 → 4 → 4b → 5 → 5b → 13 → 6 → 10 → 11 → 12 → 8a → 7a → 8b → 7b.
Documentation phases (10, 11, 12) are placed after the structural work (Phases 1–6)
so they reflect the final package layout. Phase 13 (GENIE) is placed after the
propagator registry work in Phase 5 since the registry decorator simplifies the
wiring. The last step (TFP/Flax migration) benefits most from having a complete test
suite and all other modernization already in place.
