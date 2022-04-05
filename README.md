# Hebe

Authors:

1. Carlos Argueelles Delgado
2. Christian Haack
3. Jeffrey Lazar
4. Stephan Meighen-Berger

## Table of contents

1. [Introduction](#introduction)

2. [Citation](#citation)

3. [Documentation](#documentation)

4. [Installation](#installation)


## Introduction <a name="introduction"></a>

Welcome to Hebe!

A python package to model any neutrino telescope by glueing together (and adding) different public software tools.

## Citation <a name="citation"></a>

Please cite this [software](https://github.com/Harvard-Neutrino/hebe) using
```
@software{hebe2022@github,
  author = {Carlos Argueelles Delgado and Christian Haack and Jeffrey Lazar and Stephan Meighen-Berger},
  title = {{Hebe}: Neutrino Telescope Simulation Suite,
  url = {https://github.com/Harvard-Neutrino/hebe},
  version = {0.0.1},
  year = {2022},
}
```

## Documentation <a name="documentation"></a>

The beta version of Hebe does as of yet not have any documentation. Please contact the authors if you have any problems.

## Installation <a name="installation"></a>

While you can install Hebe manually using the raw code from here, we highly recommend downloading the docker or singularity images from: [repo](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing).

Then load the docker image. In the container you would find hebe under /home/hebe/ .

We also offer a singularity image should you need it in the same [repo](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing), which may be useful for running simulations on a cluster.

For GPU usage the repository offers a GPU docker image, which you will need to use to build an image yourself. Then you will also have to compile ppc (using e.g. make gpu in the PPC_CUDA folder) youself. Note that you may change the arch version in the makefile of PPC to do this, depending on your hardware.
