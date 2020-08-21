[![Build Status](https://travis-ci.org/keeeto/super_tomo_py.svg?branch=master)](https://travis-ci.org/keeeto/super_tomo_py)
# super-resolution-ml
Code related to the `Artificial intelligence for reconstruction and super-resolution of chemical tomography` project.

## Intro

This repository contains code for using neural networks to reconstruct and enhance X-ray tomographic data. The networks in this repo are built and tested for tasks in materials science. 

## Contents

* Reconstruction
This directory contains networks that can be used to reconstruct real space images from sinographs.

## Documentation

Documentation can be found at https://superres-tomo.readthedocs.io/en/latest/index.html

## Requirements

Requirements for running the codes:
```
PIL
tensorflow >= 2.1
scikit-learn
scikit-image
h5py
matplotlib
imageio
## Only required if you want to generate sinogram shapes locally:
python 3.6
astra-toolbox
```

## Partners

The project involves [STFC](stfc.ukri.org) and [Finden Ltd.](www.finden.co.uk)

## Funding

This project is funded by the [AI3SD Network](www.ai3sd.org), reference `AI3SD-FundingCall2_017`

