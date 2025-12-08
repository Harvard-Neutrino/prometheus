# Prometheus

Welcome to Prometheus, an open-source neutrino telescope simulation.

<https://github.com/Harvard-Neutrino/prometheus>

Authors:
<!-- TODO: add IN icons? -->
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

- Python <!-- which version? -->
<!-- TODO: add Docker and anything else needed -->

## Installation

To install all of the Prometheus's dependencies, you have two options:

- download and compile them from source/`PyPI`,
- install using Docker or Singularity files.

### Install from source

#### Prerequisites

1. [LeptonInjector](https://github.com/icecube/LeptonInjector) - used to select neutrino interaction quantities. The source code for it, as well as installation guide can be found in the [project repo](https://github.com/icecube/LeptonInjector?tab=readme-ov-file#download-compilation-and-installation).

<!-- What is this step about? Install it using pip before running a setup script, or install it with alternative methods if pip doesn't work? -->
2. (optional) [PROPOSAL](https://github.com/tudo-astroparticlephysics/PROPOSAL) - used to propagate the charged leptons that result from neutrino interactions. PROPOSAL installation is included in the setup script<!-- Link to setup script -->, but people have reported issues with it in some operation systems. Therefore you can preemptively compile it at this stage either using `pip` or alternative methods, outlined in PROPOSAL's [advanced installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md).

3. (optional) [Photon propagation code (PPC)](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables) - use it for ice-based detectors simulation. There are 2 versions available:
    - regular version, runs on a CPU ([compilation instructions](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC))
    - CUDA code version, runs on a GPU ([compilation instructions](https://github.com/Harvard-Neutrino/prometheus/tree/main/resources/PPC_executables/PPC_CUDA)).

    Both of these use a modified version of the official [PPC code](https://github.com/icecube/ppc).

4. (optional) [LeptonWeighter](https://github.com/icecube/LeptonWeighter) - use it if you need to do event weighting. The source code and instructions on how to compile it can be found in the [project repo](https://github.com/icecube/LeptonWeighter?tab=readme-ov-file#installation).

#### Compilation

After installing all of the prerequisites, install Python dependencies by running the setup script in your base directory:

```sh
python setup.py install
```

If you are having issues installing PROPOSAL with `pip`, see PROPOSAL's [advanced installation guide](https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md) for alternative options.

### Using Containers

While you can install `Prometheus` manually using the raw code from here, one can also download docker or singularity image with all dependencies prebuilt from: [repo](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing).

Then load the docker image. In the container you will find prometheus under /home/myuser/prometheus . Note you may need to run bash + source /opt/.bashrc before using prometheus.

We also offer a singularity image should you need it in the same [repo](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing), which may be useful for running simulations on a cluster.
This are currently in beta and require some setup:

1. Download the latest .sif file (currently v1.0.2)
2. run ``` singularity shell name_of_file.sif ```
3. enter the /opt folder ```cd /opt ```
4. source the bash file ``` source .bashrc ```
5. clone prometheus from the repository to a folder of your choice. The setup should now be done

Please note, that some systems still use older kernels not compatible with newer boost versions. In such a case use the containers with the "old" keyword.

For GPU usage the repository offers a GPU docker image, which you will need to use to build an image yourself. Then you will also have to compile ppc (using e.g. make gpu in the PPC_CUDA folder) youself. Note that you may change the arch version in the makefile of PPC to do this, depending on your hardware.


## Citation

Please cite this [software](https://github.com/Harvard-Neutrino/prometheus) using
```
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

And please consider citing the packages that `Prometheus` uses internally, _i.e._ `LeptonInjector`, `PROPOSAL`, `PPC`, and `LeptonWeighter` with the following citations.

```
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

API documentation can be found in `/docs`

<!-- ## Installation <a name="installation"></a> -->
