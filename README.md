# Prometheus

Welcome to Prometheus, an open-source neutrino telescope simulation.

<https://github.com/Harvard-Neutrino/prometheus>

## Introduction

Prometheus is a Python-based package for simulating neutrino telescopes. Please see [the paper 2304.14526](http://arxiv.org/abs/2304.14526) for a detailed description of the package.

### Authors
<!-- TODO: check links -->
1. [Jeffrey Lazar](https://inspirehep.net/authors/1771794) 
2. [Stephan Meighen-Berger](https://inspirehep.net/authors/1828460)
3. [Christian Haack](https://inspirehep.net/authors/1284379)
4. [David Kim](https://github.com/david-kim2) <!-- TODO: is there an IN link for David ? -->
5. [Santiago Giner](https://inspirehep.net/authors/2847732)
6. [Carlos Argüelles Delgado](https://inspirehep.net/authors/1074902)

### Terms of Use

Prometheus is open-source. You are free to copy, modify, and distribute it with attribution under the terms of the GNU Lesser General Public License. See the [LICENSE](./LICENSE.md) file for details.

## Getting Started

To work with Prometheus, you will need:

- [Python](https://realpython.com/installing-python/) 3.11 or higher <!-- is this correct? -->
- [pip](https://pip.pypa.io/en/stable/installation/) for Python package management
<!-- TODO: complete list -->

Additional prerequisites depend on your installation method and are detailed in the[Installation](#installation) section below.

## Installation

There are two options to install Prometheus:

- Download and compile dependencies from source/`PyPI`
- Use provided Docker or Singularity containers with prebuilt dependencies

### Install from Source

If you have Python and pip installed, this would be the most straightforward installation method.

#### Prerequisites

Before installing Prometheus, you need to install the following packages:

1. **[LeptonInjector](https://github.com/icecube/LeptonInjector)** - Required for selecting neutrino interaction quantities. Installation instructions are available in the [project repository](https://github.com/icecube/LeptonInjector?tab=readme-ov-file#download-compilation-and-installation).

<!-- TODO: What is this step about? Install it using pip before running a setup script, or install it with alternative methods if pip doesn't work? -->
2. **[PROPOSAL](https://github.com/tudo-astroparticlephysics/PROPOSAL)** (optional) - Used to propagate charged leptons resulting from neutrino interactions. PROPOSAL installation is included in the Prometheus setup script<!-- TODO: Link to setup script -->, but some users reported issues on certain operating systems. If needed, you can install it separately, following its [installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md).

3. **[Photon Propagation Code (PPC)](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables)** (optional) - Required for the ice-based detector simulations. Two versions are available:
    - CPU version ([compilation instructions](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC))
    - GPU/CUDA version ([compilation instructions](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC_CUDA)).

    Both of these use a modified version of the official [PPC code](https://github.com/icecube/ppc).

4. **[LeptonWeighter](https://github.com/icecube/LeptonWeighter)** (optional) - Required for event weighting. Installation instructions are available in the [project repository](https://github.com/icecube/LeptonWeighter?tab=readme-ov-file#installation).

#### Compilation

After installing the prerequisites, run the setup script from the Prometheus base directory:

```sh
python setup.py install
```
<!-- TODO: Where is this setup.py located? Not finding it in the repo -->

This will install all remaining dependencies needed to run Prometheus.

> [!TIP]
> If you encounter issues installing PROPOSAL through the setup script, install it separately before running the script. See PROPOSAL's [advanced installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md) for instructions.

### Install with Containers

If source installation doesn't work for you, you can download Docker or Singularity containers with all dependencies prebuilt from [this public Google drive folder](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing).

#### Using Docker

If you need help getting started with Docker, see the [Docker documentation](https://docs.docker.com/desktop/).

After downloading the `prometheus_v_1_0_2.tar.gz` file from Google Drive, navigate to the download directory in your terminal and run:

```sh
docker load < prometheus_v_1_0_2.tar.gz
```

This will load the image. To learn more about loading images, as well as available loading options, refer to the [dockerdocs](https://docs.docker.com/reference/cli/docker/image/load/).

In the container, Prometheus is located under `/home/myuser/prometheus`.
<!-- TODO: The container doesn't run on mac M1+ (passing --platform doesn't help) - it needs better instructions + how to run the image, how to actually use the thing. -->
> [!NOTE]
> You may need to run bash and source `/opt/.bashrc` before using Prometheus within the container.

<!-- TODO: What does it mean to run bash - launch a docker shell? In what case is this needed? -->

#### Using Singularity

If you need help getting started with Singularity, see the [Singularity documentation](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html).

Singularity is particularly useful for running simulations on computing clusters. To use the Singularity container:

1. Download the latest version `.sif` file (currently v1.0.2) from the [Google Drive folder](https://drive.google.com/drive/folders/1oeI4yX_BjdStaDKzx_IsYURxSnuNY3vn) <!-- TODO: Then build the image?  -->

<!-- TODO: explain why this is needed -->
2. Launch a Singularity shell within the container:

  ```sh
  singularity shell <file_name>.sif
  ```

3. Navigate to the `/opt` directory and source the environment file:

  ```sh
  cd /opt && source .bashrc
  ```
<!-- TODO: Clone as in from git repo? Why do I need to clone it when I have a container? -->
3. Clone Prometheus from the repository to a folder of your choice.

> [!NOTE]
> Some systems still use older kernels that are not compatible with newer Boost versions. If you encounter compatibility issues, use the `.sif` files with the "old" keyword in them: `prometheus_old_<version>.sif`.

#### GPU Support
<!-- TODO: is the messaging in the new passage same as in the old one? -->

<!-- For the GPU usage the repository offers a GPU docker image, which you will need to use to build an image yourself. Then you will also have to compile PPC, e.g. using `make gpu` in the [PPC_CUDA](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC_CUDA) folder. Note that you may need to change the arch version in the makefile of PPC to do this, depending on your hardware. -->

For GPU-accelerated simulations, you will need to build a custom Docker image from the GPU Dockerfile [provided in the repository](https://github.com/Harvard-Neutrino/prometheus/tree/main/container). After building the image, compile PPC with GPU support by running `make gpu` in the [PPC_CUDA](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC_CUDA) folder.

> [!NOTE]
> You may need to modify the architecture version in the PPC makefile to match your GPU hardware.

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
  <summary>LeptonInjector and LeptonWeigher</summary>

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

## Documentation

Detailed API documentation on Prometheus' modules and classes is available in the `/docs` [directory](https://github.com/Harvard-Neutrino/prometheus/tree/main/docs/prometheus).

<!-- TODO: ## Contributing & Getting Help -->
