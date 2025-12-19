# Photon Propagation Code (ppc)

Original ppc GitHub repository: <https://github.com/icecube/ppc>

Full source code on the ppc website: <https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/>

## Downloading the code

Both versions of standalone ppc (CPU/GPU, Original/Assembly) can be downloaded from the [ppc website](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/).

On the ppc website, the red headlines in the black box at the top left link to archive files that contain the program and necessary data tables, as well as to the icetray version. Some files are also linked individually in green for quick reference.

The data tables (from one of `aha`, `spx`, or `mie` subdirectories of `dat.tgz`) need to be placed within the ppc directory before running (unless using `PPCTABLESDIR` as described below) for the CPU/GPU version, and before compilation for the Original/Assembly version.

The following notes and instructions apply only to the CPU/GPU version. They also assume that the user's shell is bash.

## Compilation

>[!CAUTION]
> The GPU version of ppc can run only on devices with compute capability 1.1 and higher. Lower compute capabilities are currently not supported.

To compile the ppc executable, run

```sh
make <VERSION>
```

Replace `<VERSION>` with the version you would like to compile: `cpu` or `gpu`.

By default only GPU devices of compute capability 1.2 or higher are supported. However, if necessary, you can compile ppc for devices of compute capability 1.1 by passing the `arch` parameter to your `make` command:

```sh
arch=11 make gpu
```

If the above compilation or running the program fails, you may need to source the `src` file. Be sure to edit this file if it does not reflect your CUDA toolkit configuration. Use the latest CUDA driver/toolkit from NVIDIA. You do not need the SDK.

Some parameters of the program can be changed via the `#define` statements in the `ini.cxx` file. Normally, there should be no need for this, as by default the program is configured for the best physics and performance with the provided data tables.

## Running the program

The basic command to run the program is:

```sh
./ppc [parameter1] [parameter2] ...
```

When run with no options, the program prints a summary of available tables. If something is missing, it prints an error.

If all necessary tables are found, the GPU version also prints a summary of the available GPU devices within your system, numbered starting from 0. These must be specified with a `[gpu]` parameter in the examples below.

The CPU version also expects this parameter. Use it to pick the random number sequence.

## Reading the program output

```sh
HIT [str] [om] [time] [costh]
```

This signifies one photon hit in OM `[str], [om]` at time `[time]` in [ns] arriving from a direction specified by $\cos({\eta})$, where $\eta$  is the angle between the PMT axis (assumed to point down) and the direction from which the photon arrived. In muon simulation, all photon hit records immediately follow the tracks on which these photons originated.

## Environment variables with relevant values

Variable | Description
----------|---------
`PPCTABLESDIR=dat/mie` | Sets the location of the data tables if not in the current directory
`WFLA=405` | Ignores the wavelength profile contained within the file `wv.dat` and simulates all photons with this single wavelength
`FLDR` | Set $FLDR=x+(n-1)\times360$, where $0\le x<360$ and $n>0$ to simulate $n$ LEDs in a symmetrical $n$-fold pattern, with first LED centered in the direction $x$. Negative or unset FLDR simulates an azimuthally symmetric pattern of light. Ignored when processing muon simulation
`BADMP` | Specifies a [bad multiprocessor (MP)](#bad-multiprocessors-mp)

### Bad multiprocessors (MP)

The `BADMP` variable allows you to run ppc on broken GPUs.

If a GPU is faulty, ppc will often exit with an `unspecified kernel failure` or a different CUDA error, hang the GPU, or crash the computer. The latter is often accompanied by a system log message (accessible with `dmesg`) such as `NVRM: Xid ...`.

For example, out of 24 GPUs in our 12 GTX 295 cards, 3 contain bad MPs. These are always the same, and are the only ones that lead to the `nan` or `inf` errors. These would be the bad MPs you want to disable.

#### Determine which MP is bad

As a rule of thumb, if ppc crashed with an error, the error message most likely includes the MP number on which the error happened. This is the MP number you should assign to the `BADMP` variable.

However, in some cases the error messages include MP numbers that do not exist on your GPU. For example, the error message includes MP #3, but your GPU has only MPs #2, 4, 5 etc.

In this case, you have two options:

1. Force-run ppc on different GPU/MP numbers.
2. Try assigning different values to the `BADMP` variable, using the MP numbers that exist on your GPU.

In the best case, your run will succeed, and in the worst case, you'll get more error messages containing information or MP numbers to try disabling.

The frequency with which bad MPs manifest appears to depend on the GPU temperature (both too high and too low, depending on the GPU) and GPU load.

## Configuration files

Below are some of the configuration files with contents and quantities used within Prometheus. For additional options, see the [ppc documentation](https://github.com/icecube/ppc).

### `cfg.txt`

A configuration file with four parameters:

- DOM radius "oversize" scaling factor
- overall DOM efficiency correction
- fraction of "SAM" contribution to the scattering function
- $g=\langle \cos(\theta) \rangle$

### `wv.dat`

Wavelength-tabulated DOM acceptance, calculated from pre-computed tables.

### `as.dat`

Parameters of the angular sensitivity polynomial expansion.

To record all arriving photons without applying the angular sensitivity curve, create the file `as.dat` with a single element set to 1:

```sh
echo 1 > as.dat
```

### `rnd.txt`

A table of random number multipliers for the multiply-with-carry random number generator. If more multipliers are needed, you can borrow the file from [CUDAMCML](https://www.atomic.physics.lu.se/biophotonics/research/monte-carlo-simulations/gpu-monte-carlo/).

### `tilt.par`, `tilt.dat`

Files describing the ice tilt, paired with particular ice models. For other ice models (AHA, SPICE 1) they must not be present.

### `icemodel.par`

A file with four parameters of the icemodel: $\alpha$, $\kappa$, $A$, $B$. Each parameter is followed by its measurement uncertainty, which is ignored by the program.

The older models (older than SPICE Lea or WHAM) have 6 parameters: $\alpha$, $\kappa$, $A$, $B$, $D$, $E$, as defined in section 3 of the [SPICE article](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/paper/a.pdf).

### `icemodel.dat`

The main ice properties table: depth of the center of the layer, $b_e(400)$, $a_{dust}(400)$, $\delta\tau$. All layers must be of equal width, and there must be at least two layers defined in the file.

If the file `icemodel.par` contains 6 parameters (as defined in section 3 of the [SPICE article](https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/paper/a.pdf)), then the absorption coefficient is calculated as:

```math
a_{dust}(400)=(D\times[\text{3rd element in a line}]+E)\times400-kappa
```
