# Installation

To work with Prometheus, you will need:

- [Python](https://realpython.com/installing-python/) 3.11 or higher
- [pip](https://pip.pypa.io/en/stable/installation/) for Python package management. It usually comes with installing Python, so you should already have it.

Additional prerequisites depend on the installation method you choose.

## Install from Source

If you have Python and pip installed, you can install Prometheus with all dependencies from source.

To start, clone this repository onto your machine:

```sh
git clone git@github.com:Harvard-Neutrino/prometheus.git
```

### Prerequisites

To work with Prometheus, you need to install the following packages:

1. **[LeptonInjector](https://github.com/icecube/LeptonInjector)** - Required for selecting neutrino interaction quantities. Installation instructions are available in the [project repository](https://github.com/icecube/LeptonInjector?tab=readme-ov-file#download-compilation-and-installation).

2. **[PROPOSAL](https://github.com/tudo-astroparticlephysics/PROPOSAL)** - Used to propagate charged leptons resulting from neutrino interactions. PROPOSAL installation is included in the Prometheus setup script, but some users reported issues on certain operating systems. If needed, you can install it separately, following its [installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md).

3. **[Photon Propagation Code (ppc)](https://github.com/icecube/ppc)** - Required for ice-based detector simulations. We use a mod  ified version of the official ppc code. Prometheus-specific compilation instructions are available in the [ppc executables directory](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables) of the Prometheus GitHub repository.

4. **[LeptonWeighter](https://github.com/icecube/LeptonWeighter)** (optional) - Required for event weighting. Installation instructions are available in the [project repository](https://github.com/icecube/LeptonWeighter?tab=readme-ov-file#installation).

### Compile

After installing the prerequisites, run the setup script from the Prometheus base directory:

```sh
python3 python -m pip setup.py install
```

This will install all remaining dependencies needed to run simulations.

!!! tip
    If you encounter issues installing PROPOSAL through the setup script, install it separately before running the script. See PROPOSAL's [advanced installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md) for instructions.

## Install with Containers

If you need to run simulations on a computing cluster or the source installation doesn't work for you, Docker and Singularity images with all dependencies prebuilt are available in [this public Google drive folder](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing).

!!! warning  
    The Docker and Singularity images we provide are not currently compatible with ARM-based system architectures, which causes issues when running them on computers with Mac M-series chips. If you are a Mac user, consider [installing Prometheus from source](#install-from-source).

### Using Docker

If you need help getting started with Docker, see the [Docker documentation](https://docs.docker.com/desktop/). Here is the outline of the steps to set things up:

1. **Load the image**

Download the latest version `.tar.gz` file from the [Google Drive folder](https://drive.google.com/drive/folders/1Ok0czMhOd8S0N73oyPn5B2mFWp_D5gUx), navigate to the directory where you downloaded the file, and run:

```sh
docker load < prometheus_v_<VERSION>.tar.gz
```

Replace `<VERSION>` with the actual version number (e.g., `1_0_2`). This will load the image into Docker. To learn more about loading images and available options, refer to the [Docker documentation](https://docs.docker.com/reference/cli/docker/image/load/).

2. **Run the container and launch a shell**

```sh
docker run -it prometheus_v_<VERSION>:latest sh
```

For more container running options, refer to the [Docker containers documentation](https://docs.docker.com/engine/containers/run/).

#### GPU Support with Docker

For GPU-accelerated simulations, you will need to build a custom Docker image from the GPU Dockerfile [provided in the repository](https://github.com/Harvard-Neutrino/prometheus/tree/main/container). After building and starting the image, compile ppc with GPU support by running `make gpu` in the [PPC_CUDA](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC_CUDA) directory within the container.

!!! note
    You may need to modify the architecture version in the ppc makefile to match your GPU hardware.

### Using Singularity

If you need help getting started with Singularity, see the [Singularity documentation](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html).

Singularity is particularly useful for running simulations on computing clusters. To use the Singularity container:

1. **Clone this repository**

The Singularity setup is still under construction, and there may be issues running software using only the files in the container. To avoid these issues, clone the repository onto your machine and navigate into the project directory:

```sh
git clone git@github.com:Harvard-Neutrino/prometheus.git && cd ./prometheus
```

2. **Download the container image**

Download the latest version `.sif` file (currently v1.0.2) from the [Google Drive folder](https://drive.google.com/drive/folders/1oeI4yX_BjdStaDKzx_IsYURxSnuNY3vn)

!!! note
    We use [Boost](https://www.boost.org/) libraries for file structure management. Some systems with older kernels may not be compatible with newer Boost versions. If you encounter compatibility issues, use the `.sif` files with the "old" keyword: `prometheus_old_<VERSION>.sif`.

3. **Launch a Singularity shell within the container**

```sh
singularity shell <FILENAME>.sif
```

Replace `<FILENAME>` with the actual name of your downloaded `.sif` file.

#### Source the Environment File

The remaining setup steps and running the software should all be done within your container shell.

Before running simulations, you may need to source the environment file to ensure all dependencies load correctly:

```sh
source /opt/.bashrc
```

Once this is done, still in your container shell, navigate into the Prometheus directory:

```sh
cd /home/myuser/prometheus
```

After that, you should be able to start running simulation scripts.
