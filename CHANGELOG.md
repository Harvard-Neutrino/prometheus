# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Fresh-install validation notes

The following was observed when running `bash install.sh --with-ppc` from a
fully clean state (no conda environment, no micromamba binary, no pre-built
packages):

**Step 1 — Environment creation (`setup_env.sh`)**
Micromamba is downloaded by `setup_env.sh` and stored in `bin/`.  This requires
an internet connection.  All 189 conda-forge packages are resolved and linked
(~5 min on first run; subsequent runs use the local `pkgs/` cache and are
near-instant).

**Step 2 — PROPOSAL (`install_proposal.sh`)**
Built from source via pip (~2 min).  This requires a working C++ tool-chain
(`g++`).  Watch for pip backtracking warnings during the prometheus editable
install step — they are harmless but can take several minutes while pip
searches for compatible versions of `flax` / `orbax-checkpoint`.

**Step 3 — LeptonInjector (`install_leptoninjector_legacy.sh`)**
Built from the vendored source at `resources/LeptonInjector/` (no network
access required). Compiled with CMake/GCC (~3 min). CMake emits a non-fatal
warning about `PYTHON_EXECUTABLE` being ignored (use `Python_ROOT_DIR`
instead) and two `CMP0074` policy warnings about `HDF5_ROOT`; these do
**not** affect the build.

**Step 4 — PPC (`install_ppc.sh`, requires `--with-ppc`)**
The symlinks `ppc.cxx → ppc.cu` and `pro.cxx → pro.cu` already exist in the
repo, so `ln -s` prints a "File exists" warning; this is harmless.

**Step 5 — Prometheus pip install (`fixes.sh`)**
Installs prometheus in editable mode and then installs `fennel-seed 2.0.0`
from the vendored source at `resources/fennel/` (no network access required).
Pip will downgrade `numpy` from 2.x to 1.26.4 and `jax`/`jaxlib` from 0.9.2
to 0.4.35 to satisfy pinned versions in `pyproject.toml`.

**First run of either example**
PROPOSAL builds its cross-section/decay tables on the first simulation run
and writes them to `resources/PROPOSAL_tables/`. This takes ~1–2 min and only
happens once. A precision warning from the PROPOSAL integrator is printed but
is purely informational. A `FutureWarning` from JAX about dtype promotion is
also non-fatal.

---

## [Unreleased] — branch `smb-version2`

### Added

- **`install.sh`** — top-level installer script that orchestrates the full
  environment setup in the correct order:
  `setup_env.sh` → `activate.sh` → `install_proposal.sh` →
  `install_leptoninjector_legacy.sh` → (optional) `install_ppc.sh` →
  `check_install.sh` → `fixes.sh`.
  Accepts `--with-ppc` flag to opt in to building the CPU PPC binary.

- **`scripts/setup_env.sh`** — downloads micromamba and creates the conda
  environment defined in `environment.yml`.

- **`scripts/activate.sh`** — activates the conda environment for a given
  environment directory.

- **`scripts/install_proposal.sh`** — installs the PROPOSAL lepton propagator
  via pip into the active environment.

- **`scripts/install_leptoninjector_legacy.sh`** — builds and installs
  LeptonInjector from the vendored source at `resources/LeptonInjector/`
  using CMake (no network access required).

- **`scripts/install_ppc.sh`** — compiles the CPU-only PPC photon propagator
  binary (`resources/PPC_executables/PPC/ppc`) from source using `g++ -O2
  --fast-math`.

- **`scripts/check_install.sh`** — verifies that all key Python packages
  (`LeptonInjector`, `EarthModelService`, `proposal`, `prometheus`) import
  without errors after installation.

- **`scripts/fixes.sh`** — runs `pip install -e .` to install prometheus in
  editable mode and installs fennel-seed 2.0.0 from the vendored source at
  `resources/fennel/`.

- **`resources/LeptonInjector/`** — vendored copy of the
  `icecube/LeptonInjector` `with_earth_py` branch (commit `d203189b`,
  Feb 2023, LGPL-3.0). All four GCC 13 / SuiteSparse 7.x compatibility
  patches are baked in. Protects against upstream branch deletion and removes
  the GitHub clone step from installation.

- **`resources/fennel/`** — vendored copy of fennel-seed 2.0.0 (commit
  `988bf2f`, MIT license). fennel-seed 2.0.0 is not published on PyPI;
  previously installed from GitHub at install time. The `notebooks/` and
  `seed/` directories are excluded as they are not needed at runtime.

- **`resources/PPC_executables/PPC/ppc`** — compiled CPU PPC binary, produced
  by `scripts/install_ppc.sh`.

- **`examples/01_basic_water.py`** — minimal end-to-end example for a water
  detector simulation.  Uses the bundled `resources/geofiles/demo_water.geo`,
  a single-event LeptonInjector volume injection of a muon neutrino CC
  interaction, and the olympus photon propagator.  Writes output to
  `output/1_photons.parquet`.

- **`examples/02_basic_ice.py`** — minimal end-to-end example for an ice
  detector simulation (PPC).  Uses the bundled
  `resources/geofiles/demo_ice.geo` and the south-pole PPC ice tables in
  `resources/PPC_tables/south_pole/`.  Runs 3 events with a volume-injected
  muon and writes output to `output/2_photons.parquet`.

### Fixed

- **`install.sh`** — added `export PATH="${PWD}/bin:${PATH}"` so the
  repo-local micromamba binary (downloaded by `setup_env.sh`) is visible to
  all sub-scripts that are invoked as separate `bash` processes.

- **`scripts/install_leptoninjector_legacy.sh`** — rewritten to build from
  `resources/LeptonInjector/` instead of cloning from GitHub. The four
  source-level GCC 13 / SuiteSparse 7.x compatibility patches (originally
  applied at build time) are now baked into the vendored tree:
  - `cmake/Packages/SuiteSparse.cmake`: replaced a `STRING(REGEX REPLACE)` on
    the entire header file content (which contains dots and newlines, breaking
    CMake 3.22 regex matching) with a `STRING(REGEX MATCH)` + conditional so
    that the major version number is extracted reliably.
  - `public/LeptonInjector/Coordinates.h`: added `#include <cstdint>` to fix
    a missing declaration of `uint8_t` under GCC 13 (which no longer includes
    `<cstdint>` transitively).
  - `public/LeptonInjector/Particle.h`: same `#include <cstdint>` fix.
  - `private/LeptonInjector/Particle.cxx`: added `#include <stdexcept>` to
    fix a missing declaration of `std::runtime_error` under GCC 13.
  - Fixed `$PYTHONPATH` unbound-variable error (triggered by `set -u`) by
    changing to `${PYTHONPATH:-}`.

- **`scripts/fixes.sh`** — updated fennel install from
  `git+https://github.com/MeighenBergerS/fennel.git@master` to the vendored
  `resources/fennel/` path.

- **`prometheus/utils/config_mims.py`** (`config_mims`): added
  `os.makedirs(os.path.dirname(output_prefix), exist_ok=True)` so that
  LeptonInjector's output directory is created automatically.  Previously the
  simulation crashed with "Failed to open … .lic" when the `output/` directory
  did not exist.

- **`prometheus/utils/config_mims.py`** (`config_mims`): fixed the call to
  `photon_prop_config_mims` — it was passing
  `config["photon propagator"][name]` (the propagator-specific sub-dict) but
  the function signature expects the full `config["photon propagator"]` dict
  so it can read the `"name"` key.

- **`prometheus/utils/config_mims.py`** (`photon_prop_config_mims`): replaced
  the `pass` stub with a real implementation.  Resolves relative `ppctables`,
  `ppc_exe`, and `ppc_tmpdir` paths in the PPC/PPC_CUDA config to absolute
  paths.  Relative paths are interpreted as relative to the `prometheus`
  package directory (`prometheus/prometheus/`) so they work regardless of the
  process working directory.
