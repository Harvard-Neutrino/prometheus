# Installation Notes

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Harvard-Neutrino/prometheus.git
cd prometheus
```

### 2. Run the installer

**Water-mode only (recommended first install):**

```bash
bash install.sh
```

**Water + ice mode (requires a C++ compiler):**

```bash
bash install.sh --with-ppc
```

The installer will:
1. Download the correct `micromamba` binary for your OS and architecture.
2. Create a self-contained conda environment at `.prometheus_env/` using `environment.yml`.
3. Install PROPOSAL (lepton energy-loss library) via pip.
4. Build and install LeptonInjector from the vendored source in `resources/LeptonInjector/`.
5. *(with `--with-ppc`)* Compile the PPC ice-photon propagator from `resources/PPC_executables/PPC/`.
6. Install the `prometheus` package itself in editable mode, plus `fennel-seed` from `resources/fennel/`.

Total time: ~10–20 minutes on first install (LI CMake build dominates).

### 3. Activate the environment

Run this in every new terminal before using Prometheus:

```bash
source scripts/activate.sh .prometheus_env
```

To avoid typing this every session, add it to your shell profile (`~/.zshrc` on macOS, `~/.bashrc` on Linux).

### 4. Run the examples

**Example 01 — water simulation (olympus/JAX, P-ONE geometry):**

```bash
python examples/01_basic_water.py
```

Runs one neutrino event through the water-Cherenkov simulation. Output is written to `output/` in the current directory (created automatically). Expected output: several hundred photon hits on the detector modules.

**Example 02 — ice simulation (PPC, IceCube geometry):**

```bash
python examples/02_basic_ice.py
```

Requires `--with-ppc` at install time and Linux (PPC is not available on macOS). Output is written to `output/` in the current directory.

---

## Supported platforms

| Platform | Status |
|---|---|
| Linux x86-64 | Fully supported |
| Linux aarch64 | Supported (micromamba download correct; LI build untested) |
| macOS arm64 (Apple Silicon) | Mostly supported — see notes below |
| macOS x86-64 (Intel) | Mostly supported — see notes below |
| Windows (native) | Not supported |
| Windows WSL2 | Same as Linux x86-64 |

---

## macOS notes

### Shell (zsh vs bash)

macOS has used **zsh** as the default shell since Catalina (10.15). The installer
detects the active shell from `$SHELL` and injects the correct micromamba hook
(`zsh` or `bash`). If you see `command not found: micromamba` after the install
completes, source the activation script explicitly in your current shell:

```zsh
source scripts/activate.sh .prometheus_env
```

### Activation hint at the end of install

The final message prints:

```
source scripts/activate.sh .prometheus_env
```

Run this in every new terminal session before using Prometheus. To make it
permanent, add it to `~/.zshrc` (or `~/.bashrc`).

### PPC (ice simulation) — not available on macOS

PPC is an IceCube GPU/CPU photon propagator that uses Linux-specific compiler flags
(`--fast-math` for GCC) and GNU make conventions. The installer skips PPC
automatically on macOS. Ice-mode simulations are not available on macOS.

Water-mode (olympus/JAX) simulations work normally.

### LeptonInjector — build may need Xcode Command Line Tools

LeptonInjector is built from source with CMake. On macOS you must have the Xcode
Command Line Tools installed:

```zsh
xcode-select --install
```

The build uses Boost and SuiteSparse from the conda environment, so no Homebrew
packages are required. However, the build has only been tested on Linux; if you
encounter build errors on macOS please open an issue with the full error output.

### `realpath -m` — fixed

The GNU `realpath -m` flag (allow non-existent paths) does not exist on the BSD
`realpath` shipped with macOS. The installer was updated to avoid calling any
`realpath` binary; path normalisation is now done in pure shell.

---

## Linux notes

The installer was developed and tested on Ubuntu 22.04 / 24.04 (x86-64). It should
work on any modern Debian/Ubuntu or RHEL-based distribution. `curl` must be
available (`apt install curl` / `dnf install curl`).

---

## Known limitations

| Issue | Workaround |
|---|---|
| PPC not available on macOS | Water-mode simulations (olympus) work without PPC |
| LeptonInjector build on macOS is untested | Report build failures with full CMake output |
| GENIE loading requires optional deps | `pip install uproot pandas` or see Phase 13 of refactor.md |
| Intel Arc GPU not accessible in WSL2 | Requires `intel-compute-runtime` inside WSL2 and a recent Windows Intel Graphics driver; see docs/prometheus/gpu.md (planned) |
