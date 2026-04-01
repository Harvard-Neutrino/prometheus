# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

- **`scripts/install_leptoninjector_legacy.sh`** — clones the
  `with_earth_py` branch of LeptonInjector from GitHub, applies source-level
  compatibility patches (see Fixed below), and builds/installs it with CMake.

- **`scripts/install_ppc.sh`** — compiles the CPU-only PPC photon propagator
  binary (`resources/PPC_executables/PPC/ppc`) from source using `g++ -O2
  --fast-math`.

- **`scripts/check_install.sh`** — verifies that all key Python packages
  (`LeptonInjector`, `EarthModelService`, `proposal`, `prometheus`) import
  without errors after installation.

- **`scripts/fixes.sh`** — runs `pip install -e .` to install prometheus in
  editable mode and applies any remaining post-build fixups.

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

- **`scripts/install_leptoninjector_legacy.sh`** — four source-level patches
  applied at build time to make LeptonInjector compile against GCC 13 and
  SuiteSparse 7.x:
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
