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
export PPC_EXE="${PWD}/resources/PPC_executables/PPC/ppc"
export PPC_TABLES_DIR=/path/to/ice/tables
```

`PPC_EXE` is built by `install.sh --with-ppc` and lives at `resources/PPC_executables/PPC/ppc`.

Ice tables are not bundled with Prometheus; download them from the [ppc website](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/). Unpack one of the `aha`, `spx`, or `mie` subdirectories from `dat.tgz` and point `PPC_TABLES_DIR` at it. The directory must contain at minimum `cfg.txt`, `icemodel.dat`, `icemodel.par`, `wv.dat`, `as.dat`, and `rnd.txt`.

### 4. PPC for multi-PMT / next-gen modules

The multi-PMT (next-gen) mode — used by the DEgg, WOM, and mDOM-style examples — requires a version of PPC that supports `om.conf` and `om.map`. This support lives in the upstream PPC repository at:

**<https://github.com/icecube/ppc>**

It is included as a git submodule at `resources/PPC_executables/icecube-ppc`, pinned to `V1.0.1` (commit `bdaa599`, 2025-10-02). Fetch and build it with:

```bash
git submodule update --init resources/PPC_executables/icecube-ppc
bash scripts/install_ppc.sh
export PPC_EXE="${PWD}/resources/PPC_executables/icecube-ppc/gpu/ppc"
```

#### PPC version setting

Set `ppc_version` to `"official"` whenever `ppc_exe` is the official PPC, also for detectors without multi-PMT modules. The default is `"bundled"`, for the copies in `PPC/` and `PPC_CUDA/`. The two read track angles and write hit angles differently, and a wrong setting gives wrong directions. Multi-PMT modules raise an error unless it is `"official"`.

#### Module tables

PPC needs one `om.wv_<type>.<pmt type>` file per module type. The LOM-16 in `resources/optical_modules/lom16` has one; tables for other modules are not distributed with Prometheus. Put yours in `ppctables`, or in a directory given as `om_tables`.

- Each `om.wv_*` line is wavelength [nm] then direction-averaged effective area [cm^2]. The PPC documentation states the opposite order, but the PPC code reads wavelength first.
- Prometheus raises an error if a table is missing, because PPC would otherwise return no hits for that module type.
- `prometheus.detector.read_om_conf` turns an `om.conf` file into `Module` arguments. The type IDs must match the table names.
- `om.dirs` is taken from `om_dirs` or `ppctables` if present, and generated otherwise.
- Remove IceCube's per-DOM files (`eff-f2k`, `dx.dat`) from `ppctables`; PPC would apply them to modules with the same string and module number.

#### Multi-PMT detectors in a full simulation

This is experimental. Geometry files cannot describe module types, so build the detector in Python:

```python
from prometheus import Prometheus, config
from prometheus.detector import Detector, Medium, Module, read_om_conf

lom16 = read_om_conf("resources/optical_modules/lom16/om.conf")[171]
modules = [Module(position, key, **lom16) for position, key in zip(positions, keys)]

config.photon_propagator.name = "PPC"
ppc = config.photon_propagator.ppc
ppc.paths.ppc_exe = "/path/to/icecube-ppc/gpu/ppc"
ppc.paths.ppctables = "/path/to/ice/tables"
ppc.paths.om_tables = "resources/optical_modules/lom16"
ppc.simulation.ppc_version = "official"
ppc.simulation.output_mode = "extended"
Prometheus(config, detector=Detector(modules, Medium.ICE)).sim()
```

#### LOM example

```bash
python examples/05_lom_ppc.py --out examples/output/lom16.parquet
python examples/04_event_view.py examples/output/lom16.parquet --om-conf resources/optical_modules/lom16/om.conf --om-type 171
```

To simulate a muon neutrino in a multi-string array and look at it in a browser:

```bash
python examples/07_numu_lom_array.py
python examples/08_event_display.py examples/output/numu_lom/1_photons.parquet --geo examples/output/numu_lom/detector.geo
```

`07_numu_lom_array.py --n-events 100` simulates random events; the display then has a menu and arrow keys to move between them. On a CPU this takes about 20 minutes.

`06_fit_lom_shape.py` derives the LOM-16 values from public sources. Its inputs and citations are in `data/lom/README.md`.

PPC assigns each hit to a PMT from the photon direction only, not from where the photon lands on the module, and gives every PMT the same angular response.

The copies in `resources/PPC_executables/PPC/` and `PPC_CUDA/` are older and do not support multi-PMT modules.

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
| `05_lom_ppc.py` | IceCube-Gen2 LOM simulation with the official PPC |
| `06_fit_lom_shape.py` | Fit the LOM-16 `om.conf` values to its published effective area |
| `07_numu_lom_array.py` | Simulate a muon neutrino in a multi-string LOM-16 array |
| `08_event_display.py` | Write an interactive 3-D display of a multi-PMT event as one HTML file |
| `09_gen1_dom_comparison.py` | Compare a one-PMT multi-PMT module with the vanilla PPC DOM |
| `genie/01_genie_injection.py` | Neutrino injection using a GENIE flux |

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
