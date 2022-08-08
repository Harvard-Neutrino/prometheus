# Hebe

Authors:

1. Jeffrey Lazar
2. Stephan Meighen-Berger
3. Christian Haack
4. David Kim
5. Carlos Argueelles Delgado


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
  author = {Jeffrey Lazar and Stephan Meighen-Berger and Christian Haack and David Kim and Carlos Argueelles Delgado},
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

Then load the docker image. In the container you would find hebe under /opt/hebe/ .

We also offer a singularity image should you need it in the same [repo](https://drive.google.com/drive/folders/1-PbSiZQr0n85g9PrhbHMeURDOA02QUSY?usp=sharing), which may be useful for running simulations on a cluster.
This are currently in beta and require some setup:

1. Download the latest .sif file (currently v3.0.3)
2. run ``` singularity shell name_of_file.sif ```
3. enter the /opt folder ```cd /opt ```
4. source the bash file ``` source .bashrc ```
5. clone hebe from the repository to a folder of your choice. The setup should now be done

For GPU usage the repository offers a GPU docker image, which you will need to use to build an image yourself. Then you will also have to compile ppc (using e.g. make gpu in the PPC_CUDA folder) youself. Note that you may change the arch version in the makefile of PPC to do this, depending on your hardware.
