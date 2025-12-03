# Prometheus

Welcome to Prometheus!

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

`Prometheus` is a `Python`-based package for simulating neutrino telescopes. Please see [2304.14526](http://arxiv.org/abs/2304.14526) for a detailed description of the package.

## Dependencies

`Prometheus` depends a few external packages, one can either install these manually or download images with the dependencies prebuilt.
We provide details for providing either option here.

## Installation

To install all of `Prometheus`'s dependencies one has two options: either by downloading and compiling them from `PyPI`or source, or by downloading Docker or Singularity files.

### Installation from source

First, we will discuss the packages which must be compiled from source.

First, we must install `LeptonInjector` which will be used to select neutrino interaction quantities.
The source code for this as well as installation instructions can be found [here](https://github.com/icecube/LeptonInjector).

Next, one may optionally install `PROPOSAL`, which is used to propagate the charged leptons that result from neutrino interactions.
We say this is optional since it can also be installed using the setup script later.
We only mention compiling it from source here because there have occasionally been issues with using `pip` for some operating systems.

Next, if one wishes to simulate ice-based detectors, it is necessary to compile the photon propagation code.
The source code for this can be found in `/resources/PPC_executables/`.
We provide two directories here, one containing code suitable for running on a CPU and one containing CUDA code that can run on a GPU.
Details on compiling them can be found in those directories.
These use a modified version of the official PPC code found [here](https://github.com/icecube/ppc).

Lastly, if one wishes to do event weighting, one must compile `LeptonWeighter`.
The source code and instructions on how to compile it can be found [here](https://github.com/icecube/LeptonWeighter).

After this, one can install all the Python dependencies by running the setup script found in the base directory of the repo, _i.e._ by running `python setup.py install` from the base directory.
There have been some issues with installing `PROPOSAL` via pip.
If this is encoutered, please see instructions for installing from source above.

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

API documentation can be found in `/documentation`

<!-- ## Installation <a name="installation"></a> -->
