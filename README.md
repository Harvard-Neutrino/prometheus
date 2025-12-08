# Prometheus

Welcome to Prometheus, an open-source neutrino telescope simulation.

<https://github.com/Harvard-Neutrino/prometheus>

Authors:
<!-- TODO: add IN icons or format this nicer -->
1. [Jeffrey Lazar](https://inspirehep.net/authors/1771794)
2. [Stephan Meighen-Berger](https://inspirehep.net/authors/1828460)
3. [Christian Haack](https://inspirehep.net/authors/1284379)
4. [David Kim](https://github.com/david-kim2)
5. [Santiago Giner](https://inspirehep.net/authors/2847732)
6. [Carlos Argüelles Delgado](https://inspirehep.net/authors/1074902)

## Introduction

Prometheus is a Python-based package for simulating neutrino telescopes. Please see [the paper 2304.14526](http://arxiv.org/abs/2304.14526) for a detailed description of the package.

## Contributions

Prometheus is open-source. You are free to copy, modify, and distribute it with attribution under the terms of the GNU Lesser General Public License. See the LICENSE file for details.

## Getting Started

To work with Prometheus, you will need:

- Python 3.11 or higher <!-- is this correct? -->
- pip
<!-- TODO: complete list -->

Other prerequisits depend on how you choose to install the package. More details are available in [Installation](#installation) section below.

## Installation

To install all of the Prometheus's dependencies, you have two options:

- download and compile them from source/`PyPI`,
- install using Docker or Singularity files.

### Install from source

If you have Python and pip installed, this would be the easiest way to get you going.

#### Prerequisites

Before installing dependencies and using Prometheus, you will need to install/compile those libraries first:

1.[LeptonInjector](https://github.com/icecube/LeptonInjector) - used to select neutrino interaction quantities. The source code for it, as well as installation guide can be found in the [project repo](https://github.com/icecube/LeptonInjector?tab=readme-ov-file#download-compilation-and-installation).

<!-- TODO: What is this step about? Install it using pip before running a setup script, or install it with alternative methods if pip doesn't work? -->
2.(optional) [PROPOSAL](https://github.com/tudo-astroparticlephysics/PROPOSAL) - used to propagate the charged leptons that result from neutrino interactions. PROPOSAL installation is included in the setup script<!-- TODO: Link to setup script -->, but people have reported issues with it in some operation systems. Therefore you can preemptively compile it at this stage either using `pip` or alternative methods, outlined in PROPOSAL's [advanced installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md).

3.(optional) [Photon propagation code (PPC)](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables) - use it for ice-based detectors simulation. There are 2 versions available:
    - regular version, runs on a CPU ([compilation instructions](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC))
    - CUDA code version, runs on a GPU ([compilation instructions](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC_CUDA)).

    Both of these use a modified version of the official [PPC code](https://github.com/icecube/ppc).

4.(optional) [LeptonWeighter](https://github.com/icecube/LeptonWeighter) - use it if you need to do event weighting. The source code and instructions on how to compile it can be found in the [project repo](https://github.com/icecube/LeptonWeighter?tab=readme-ov-file#installation).

#### Compilation

After installing all of the prerequisites, install Python dependencies by running the setup script in your base directory:

```sh
  python setup.py install
```

If you are having issues installing PROPOSAL with `pip`, see PROPOSAL's [advanced installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md) for alternative options.

### Install with Containers

If installing from source doesn't work for you, feel free to download a Docker or Singularity image with all dependencies already prebuilt from [this public Google drive folder](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing).

#### Using Docker

If you don't know what Docker is, or don't have it installed, instructions in [dockerdocs](https://docs.docker.com/desktop/) can help you get started.

After downloading the `prometheus_v_1_0_2.tar.gz` file from Google drive folder, navigate to the directory where your file is located in your terminal and run this command to [load the image](https://docs.docker.com/reference/cli/docker/image/load/):

```sh
  docker load < prometheus_v_1_0_2.tar.gz
```

In the container you will find Prometheus under `/home/myuser/prometheus`.
<!-- TODO: The container doesn't run on mac M1+ (passing --platform doesn't help) - it needs better instructions + how to run the image, how to actually use the thing. -->

Note that you may need to run bash + `source /opt/.bashrc` before using Prometheus.
<!-- TODO: What does it mean to run bash - launch a CLI? In what case is this needed? -->

#### Using Singularity

If you don't know what Singularity is, or don't have it installed, [Singularity docs](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) can help you get started.

Using Singularity is helpful for performing simulations on a cluster. It's a beta version setup which requires some extra steps to run:

1.Download the latest version `.sif` file (currently v1.0.2)

2.In your terminal, run these commands:
<!-- TODO: would be nice to explain what the steps do -->
```sh
# enter singularity shell
singularity shell <file_name>.sif
# navigate into the /opt directory
cd /opt
# source the bash file
source .bashrc
```
<!-- TODO: Clone as in from git repo? -->
3.Clone prometheus from the repository to a folder of your choice.

<!-- TODO: How to verify the setup worked? -->
Please note, that some systems still use older kernels not compatible with newer boost versions. If that's your case, you can download the `.sif` files with the "old" keyword in them: `prometheus_old_<version>.sif`.

#### Using GPU
<!-- TODO: unedited, need to clarify what this means and  where this docker image is -->

For the GPU usage the repository offers a GPU docker image, which you will need to use to build an image yourself. Then you will also have to compile PPC (using e.g. make gpu in the PPC_CUDA folder) youself. Note that you may change the arch version in the makefile of PPC to do this, depending on your hardware.

## Citation

Please cite this software like so:

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

Also, please consider citing the packages that `Prometheus` uses internally: `LeptonInjector`, `PROPOSAL`, `PPC`, and `LeptonWeighter` with the following citations:

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

## Documentation
<!-- TODO: remove this when the doc site is up -->

Detailed API documentation on Prometheus' modules and classes is available in the [/docs directory](https://github.com/Harvard-Neutrino/prometheus/tree/main/docs/prometheus).
