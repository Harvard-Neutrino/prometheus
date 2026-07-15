# Prometheus Examples

This directory contains example scripts and notebooks demonstrating how to use Prometheus.

## Setup

### 1. Install

From the repository root, run the installer. Pass `--with-ppc` to also compile the PPC photon propagator (required for simulation runs):

```bash
bash install.sh --with-ppc
```

If you only want to explore the module API and serialization without running simulations, `--with-ppc` is optional.

### 2. Activate the environment

```bash
source scripts/activate.sh .prometheus_env
```

Run this every time you open a new shell before using Prometheus.

### 3. PPC environment variables (simulation only)

Cells or scripts that call the photon propagator need two environment variables pointing at the compiled PPC binary and the ice tables:

```bash
export PPC_EXE="${PWD}/resources/PPC_executables/bin/ppc_cpu"
export PPC_TABLES_DIR=/path/to/ice/tables
```

`PPC_EXE` is built by `install.sh --with-ppc` and lives at `resources/PPC_executables/bin/ppc_cpu` (CPU) or `bin/ppc_gpu` (CUDA).

Ice tables are not bundled with Prometheus; download them from the [ppc website](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/). Unpack one of the `aha`, `spx`, or `mie` subdirectories from `dat.tgz` and point `PPC_TABLES_DIR` at it. The directory must contain at minimum `cfg.txt`, `icemodel.dat`, `icemodel.par`, `wv.dat`, `as.dat`, and `rnd.txt`.

### 4. PPC for multi-PMT / next-gen modules

The multi-PMT (next-gen) mode — used by the DEgg, WOM, and mDOM-style examples — requires a version of PPC that supports `om.conf` and `om.map`. This support lives in the upstream PPC repository at:

**<https://github.com/icecube/ppc>**

The engine is vendored as a git submodule at `resources/PPC_executables/ppc-src/`
(pinned to a specific `icecube/ppc` commit). `install.sh --with-ppc` builds it
into `resources/PPC_executables/bin/`. To build manually:

```bash
git submodule update --init resources/PPC_executables/ppc-src
cd resources/PPC_executables/ppc-src/gpu
make cpu        # CPU-only build (-> ./ppc)
# or: make gpu  # CUDA build (requires NVIDIA toolkit)
```

See `resources/PPC_executables/README.md` for full build and configuration details, including GPU compute capability requirements and the `PPCTABLESDIR` environment variable.

## Examples

### Basic simulation scripts

The numbered scripts each demonstrate a self-contained simulation. Run them from the `examples/` directory after activating the environment and setting the PPC env vars:

| Script | Description |
|--------|-------------|
| `01_basic_water.py` | Single-string detector in water |
| `02_basic_ice.py` | Single-string detector in ice |
| `03_docker_water.py` | Water simulation using a Docker-based PPC image |
| `04_event_view.py` | Load and visualise simulated events |
| `05_genie_injection.py` | Neutrino injection using a GENIE flux |

```bash
cd examples
python 01_basic_water.py
```

Config files (`water_config.toml`, `ice_config.toml`) in this directory are used by the scripts and can be edited to change detector geometry, particle type, and output paths.

### Multi-PMT optical module notebook

`multi_pmt_om_example.ipynb` walks through building next-generation multi-PMT modules (DEgg, WOM, mDOM-style), generating PPC config files, running photon propagation, and reading per-PMT hits.

```bash
cd examples
jupyter notebook multi_pmt_om_example.ipynb
```

Most cells (module construction, `om.conf`/`om.map` generation, hit parsing, serialization) run without PPC. Only the propagation cell (section 5) requires `PPC_EXE` and `PPC_TABLES_DIR`; it prints a message and skips gracefully if they are not set.

Key topics covered in the notebook:

- **Module shapes:** sphere (IceCube DOM), spheroid (DEgg), cylinder (WOM) via `Rr`/`Rz`
- **PMT directions:** arbitrary `(zenith_deg, azimuth_deg)` per PMT via `pmt_dirs`
- **Angular acceptance:** the `beta` parameter written to `om.conf`
- **Mixed detectors:** legacy (`module_type=-1`) and next-gen modules in the same run
- **Hit serialization:** `minimal`, `standard`, and `extended` output modes, including per-PMT `pmt_id` and Cartesian hit position (`hit_x/y/z`)
