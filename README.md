# Prometheus

Welcome to Prometheus, an open-source neutrino telescope simulation.

<https://github.com/Harvard-Neutrino/prometheus>

## Introduction

Prometheus is a Python-based package for simulating neutrino telescopes. It balances ease of use with performance and supports simulations with arbitrary detector geometries deployed in ice or water.

Prometheus simulates neutrino interactions in the volume surrounding the detector, computes the light yield of hadronic showers and the outgoing leptons, propagates photons in the medium, and records their arrival times and positions in user-defined regions. Events are then serialized to Parquet files, a compact and interoperable format that enables efficient access for downstream analysis.

For a detailed description of the package, see [arXiv:2304.14526](http://arxiv.org/abs/2304.14526).

### Authors

1. [Jeffrey Lazar](https://inspirehep.net/authors/1771794) 
2. [Stephan Meighen-Berger](https://inspirehep.net/authors/1828460)
3. [Christian Haack](https://inspirehep.net/authors/1284379)
4. [David Kim](https://github.com/david-kim2)
5. [Santiago Giner](https://inspirehep.net/authors/2847732)
6. [Carlos Argüelles Delgado](https://inspirehep.net/authors/1074902)

### Terms of Use

Prometheus is open-source. You are free to copy, modify, and distribute it with attribution under the terms of the GNU Lesser General Public License. See the [LICENSE](./LICENSE.md) file for details.

## Getting Started

To work with Prometheus, you will need:

- [Python](https://realpython.com/installing-python/) 3.11 or higher
- [pip](https://pip.pypa.io/en/stable/installation/) for Python package management

Additional prerequisites depend on your installation method and are detailed in the [Installation](#installation) section below.

## Installation

There are two options to install Prometheus:

- Download and compile dependencies from source/`PyPI`
- Use provided Docker or Singularity containers with prebuilt dependencies

### Install from Source

If you have Python and pip installed, this would be the most straightforward installation method.

To start, clone this repository onto your machine:

```sh
git clone git@github.com:Harvard-Neutrino/prometheus.git
```

#### Prerequisites

To work with Prometheus, you need to install the following packages:

1. **[LeptonInjector](https://github.com/icecube/LeptonInjector)** - Required for selecting neutrino interaction quantities. Installation instructions are available in the [project repository](https://github.com/icecube/LeptonInjector?tab=readme-ov-file#download-compilation-and-installation).

2. **[PROPOSAL](https://github.com/tudo-astroparticlephysics/PROPOSAL)** (optional) - Used to propagate charged leptons resulting from neutrino interactions. PROPOSAL installation is included in the Prometheus setup script, but some users reported issues on certain operating systems. If needed, you can install it separately, following its [installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md).

3. **[Photon Propagation Code (PPC)](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables)** (optional) - Required for ice-based detector simulations. Two versions are available:
    - CPU version ([compilation instructions](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC))
    - GPU/CUDA version ([compilation instructions](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC_CUDA)).

    Both of these use a modified version of the official [PPC code](https://github.com/icecube/ppc).

4. **[LeptonWeighter](https://github.com/icecube/LeptonWeighter)** (optional) - Required for event weighting. Installation instructions are available in the [project repository](https://github.com/icecube/LeptonWeighter?tab=readme-ov-file#installation).

#### Compilation

> [!IMPORTANT]  
> The Prometheus setup script is currently under review to ensure its steps are up to date. We are working to make it available as soon as possible.

After installing the prerequisites, run the setup script from the Prometheus base directory:

```sh
python setup.py install
```

This will install all remaining dependencies needed to run simulations.

> [!TIP]
> If you encounter issues installing PROPOSAL through the setup script, install it separately before running the script. See PROPOSAL's [advanced installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md) for instructions.

### Install with Containers

If you need to run simulations on a computing cluster or the source installation doesn't work for you, Docker and Singularity images with all dependencies prebuilt are available in [this public Google drive folder](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing).

> [!WARNING]  
> The Docker and Singularity images we provide are not currently compatible with ARM-based system architectures, which causes issues when running them on computers with Mac M-series chips. If you are a Mac user, consider [installing Prometheus from source](#install-from-source).

#### Using Docker

If you need help getting started with Docker, see the [Docker documentation](https://docs.docker.com/desktop/). Here is the outline of the steps you need to take to set things up:

1. Load the image

Download the latest version `.tar.gz` file from the [Google Drive folder](https://drive.google.com/drive/folders/1Ok0czMhOd8S0N73oyPn5B2mFWp_D5gUx), navigate to the directory where you downloaded the file, and run:

```sh
docker load < prometheus_v_<VERSION>.tar.gz
```

Replace `<VERSION>` with the actual version number (e.g., `1_0_2`). This will load the image into Docker. To learn more about loading images and available options, refer to the [Docker documentation](https://docs.docker.com/reference/cli/docker/image/load/).

2. Run the container and launch a shell:

```sh
docker run -it prometheus_v_<VERSION>:latest sh
```

For more container running options, refer to the [Docker containers documentation](https://docs.docker.com/engine/containers/run/).

#### GPU Support with Docker

For GPU-accelerated simulations, you will need to build a custom Docker image from the GPU Dockerfile [provided in the repository](https://github.com/Harvard-Neutrino/prometheus/tree/main/container). After building and starting the image, compile PPC with GPU support by running `make gpu` in the [PPC_CUDA](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC_CUDA) directory within the container.

>[!NOTE]
>You may need to modify the architecture version in the PPC makefile to match your GPU hardware.

#### Using Singularity

If you need help getting started with Singularity, see the [Singularity documentation](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html).

Singularity is particularly useful for running simulations on computing clusters. To use the Singularity container:

1. **Clone this repository**

The Singularity setup is still under construction, and there may be issues running software using only the files in the container. To avoid these issues, clone the repository onto your machine and navigate into the project directory:

```sh
git clone git@github.com:Harvard-Neutrino/prometheus.git && cd ./prometheus
```

2. **Download the container image**

Download the latest version `.sif` file (currently v1.0.2) from the [Google Drive folder](https://drive.google.com/drive/folders/1oeI4yX_BjdStaDKzx_IsYURxSnuNY3vn)

> [!NOTE]
> We use [Boost](https://www.boost.org/) libraries for file structure management. Some systems with older kernels may not be compatible with newer Boost versions. If you encounter compatibility issues, use the `.sif` files with the "old" keyword: `prometheus_old_<VERSION>.sif`.

3. **Launch a Singularity shell within the container**

```sh
singularity shell <FILENAME>.sif
```

Replace `<FILENAME>` with the actual name of your downloaded `.sif` file.

#### Source the Environment File

> [!TIP]
> The remaining setup steps and running the software should all be done within your container shell.

Before running simulations, you may need to source the environment file to ensure all dependencies load correctly:

```sh
source /opt/.bashrc
```

Once this is done, still in your container shell, navigate into the Prometheus directory:

```sh
cd /home/myuser/prometheus
```

After that, you should be able to start running simulation scripts.

## Running Simulations

The [examples directory](https://github.com/Harvard-Neutrino/prometheus/tree/ana/update-readme/examples) contains example scripts for running simulations.

As a first-time user, you will need `example_ppc.py` for ice-based simulations and `example_olympus.py` for water-based ones. Other scripts cover more specific use case scenarios.

To execute a script, run:

```sh
python <SCRIPT_NAME>.py
```

Replace `<SCRIPT_NAME>` with the name of your simulation script.

To learn more about how Prometheus scripts work, refer to the API documentation in the [docs directory](https://github.com/Harvard-Neutrino/prometheus/tree/main/docs/prometheus).

## Citation

Please cite Prometheus using this entry:

```bibtex
@article{Lazar:2023rol,
    author = {Lazar, Jeffrey and Meighen-Berger, Stephan and Haack, Christian and Kim, David and Giner, Santiago and Arg\"uelles, Carlos A.},
    title = "{Prometheus: An Open-Source Neutrino Telescope Simulation}",
    eprint = "2304.14526",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    month = "4",
    year = "2023"
}
```

Please also consider citing the packages that Prometheus uses internally: LeptonInjector, PROPOSAL, PPC, and LeptonWeighter with the following citations:

<details>
  <summary>LeptonInjector and LeptonWeighter</summary>

  ```bibtex
  @article{IceCube:2020tcq,
      author = "Abbasi, R. and others",
      collaboration = "IceCube",
      title = "{LeptonInjector and LeptonWeighter: A neutrino event generator and weighter for neutrino observatories}",
      eprint = "2012.10449",
      archivePrefix = "arXiv",
      primaryClass = "physics.comp-ph",
      doi = "10.1016/j.cpc.2021.108018",
      journal = "Comput. Phys. Commun.",
      volume = "266",
      pages = "108018",
      year = "2021"
  }
  ```
</details>

<details>
  <summary>PROPOSAL</summary>

  ```bibtex
  @article{koehne2013proposal,
      title     = {PROPOSAL: A tool for propagation of charged leptons},
      author    = {Koehne, Jan-Hendrik and
                  Frantzen, Katharina and
                  Schmitz, Martin and
                  Fuchs, Tomasz and
                  Rhode, Wolfgang and
                  Chirkin, Dmitry and
                  Tjus, J Becker},
      journal   = {Computer Physics Communications},
      volume    = {184},
      number    = {9},
      pages     = {2070--2090},
      year      = {2013},
      doi       = {10.1016/j.cpc.2013.04.001}
  }
  ```

</details>

<details>
  <summary>PPC</summary>

  ```bibtex
  @misc{chirkin2022kpl,
      author = {D. Chirkin},
      title = {ppc},
      year = {2022},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/icecube/ppc}},
      commit = {30ea4ada13fbcf996c58a3eb3f0b1358be716fc8}
  }

  ```

</details>

## API Documentation

Detailed API documentation on Prometheus' modules and classes is available in the [docs directory](https://github.com/Harvard-Neutrino/prometheus/tree/main/docs/prometheus).

## Getting Help

If you encounter any issues with installation, setup, or bugs in this software, feel free to open a GitHub issue and we will address it as soon as we can.

If you are completely blocked and we cannot get to your issue in time, or if you have questions or suggestions about the software, please feel free to contact the authors using the email addresses listed on Inspire-HEP.
