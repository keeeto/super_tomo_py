- [1 Project Details](#1-project-details)
- [2 Project Team](#2-project-team)
  * [2.1 Principal investigator](#21-principal-investigator)
  * [2.2 Co-investigators](#22-co-investigators)
- [3. Publicity summary](#3-publicity-summary)
- [4 Executive summary](#4-executive-summary)
- [5 Aims and objectives](#5-aims-and-objectives)
- [6 Methodology](#6-methodology)
  * [6.1 Scientific methodology](#61-scientific-methodology)
  * [6.2 AI methodology](#62-ai-methodology)
- [7 Interim results](#7-interim-results)
  * [7.1 Libraries](#71-libraries)
  * [7.2 CNN tomographic reconstruction](#72-cnn-tomographic-reconstruction)
  * [7.3 MSDN for reconstruction/super-resolution](#73-msdn-for-reconstruction-super-resolution)
- [8 Outputs](#8-outputs)
  * [8.1 Libraries](#81-libraries)
  * [8.2 Networks](#82-networks)
- [9 Progress Summary](#9-progress-summary)
- [10 Next steps](#10-next-steps)

[Table of contents generated with markdown-toc](http://ecotrust-canada.github.io/markdown-toc/)


## 1 Project Details

|                   |                                                              |
| ----------------- | ------------------------------------------------------------ |
| Title             | Artificial intelligence for reconstruction and super-resolution of chemical tomography |
| Funding reference | AI3SD-FundingCall2_017                                       |
| Lead institution  | STFC                                                         |
| Project dates     | 01/05/2020 - 31/10/2020                                      |
| Website           | [Project](https://superres-tomo.readthedocs.io/en/latest/about.html) [SciML STFC](https://www.scd.stfc.ac.uk/Pages/Scientific-Machine-Learning.aspx) [Finden](www.finden.co.uk) |
| Keywords          | Chemical tomography, machine learning, CNNs, GANs            |

## 2 Project Team

### 2.1 Principal investigator

**Name and Title:** Dr Keith Butler (Data Scientist)
**Association:** STFC, Scientific Computing Department
**Work Email:** keith.butler@stfc.ac.uk
**Work Phone:** 01925 606541

### 2.2 Co-investigators

**Name and Title:** Mr Honyang Dong (PhD student)
**Association:** University College London and Finden Ltd
**Work Email:** hongyang.dong.18@ucl.ac.uk
**Work Phone:** 01235 56 7397

**Name and Title:** Dr Antony Vamvakeros (Research Scientist)
**Association:** Finden Ltd
**Work Email:** antony@finden.co.uk
**Work Phone:** 01235 56 7397

**Name and Title:** Dr Simon Jacques (Managing Director)
**Association:** Finden Ltd
**Work Email:** simon@finden.co.uk
**Work Phone:** 01235 56 7397

## 3. Publicity summary

X-ray scatter-based tomography allows unprecedented insight into the chemical and physical state of functional materials and devices. Such tomographies can be used as research tools but also offer the prospect of routine scanning for security and inspection systems and potential for medical scanning. However X-ray scatter tomogrpahy requires longer collection times and higher doses than conventional absorption tomography - in this project we will develop machine learning tools to fuse X-ray scatter and X-ray absorption tomogrpahy, providing the detail of the former with the efficiency of the latter.

In conventional X-ray tomography, the images that are obtained give maps of density within the object and the composing pixels contain single grey scale values. In scatter based tomography, each pixel instead contains spectrum or equivalent chemical signal i.e. a 1D array (or higher) of numbers. An X-ray scatter tomography slice becomes a data cube with the two conventional spatial dimensions and a third spectral dimension. Such image data is termed hyperspectral.   

![alternate text](https://superres-tomo.readthedocs.io/en/latest/_images/cnn-reconstruct.png)*Figure 1 A neural network converts a sinogram (left), into a real-space image (right).*

Whilst hyperspectral tomography can match or even exceed the resolution offered by conventional X-ray absorption contrast tomography, the latter is more highly optimised and offers modalities that can generate images in a fraction of the time and dose to image the same volume. In practice it is often the case that hyperspectral tomography resolution is sacrificed to accelerate collection time. This project aims to exploit machine learning approaches to marry hyperspectral chemical tomography with conventional X-ray absorption tomography to achieve chemical images with the rich information of the former in combination with the resolution and speed of collection of the latter.

## 4 Executive summary

X-ray scatter based scatter tomography are extremely powerful non-destructive analytical techniques that can provide insight into the chemical and physical states of functional materials and devices even under operating conditions. These approaches have potential for applications beyond their current use as research tools, but to date this has not been realised in large due to the long collection times and high dose rates associated with measurement. The aim in this project is to facilitate shorter data collections that can still yield high-resolution images by using machine learning approaches to achieve super-resolution. To achieve this we will use the information from the large number of data points within the hyperspectral X-ray scatter dataset and combine with the traditional conventional X-ray absorption signal which can be easily and quickly measured. This project will focus on (1) developing and applying novel AI-based methods for chemical image (volume) reconstruction and (2) enhancing the spatial resolution of the chemical images. 

We will be applying the approaches to X-ray diffraction computed tomography technique. The state-of-the-art with this method, can yield large 3D volumes, containing many hundreds of thousands or even millions of diffraction patterns. These are extremely challenging and time consuming to processes and analyse. The continuing development in instrumentation means that this big data problem is only increasing and indeed this problem becomes even worse when higher resolution chemical images are obtained.

The new machine learning based algorithms that are being developed will automate or semi-automate the image reconstruction process while the image enhancement will allow the collection of less data to achieve the same resolution (both cases falling under the big data handling umbrella). These new reconstruction and image enhancement approaches will be immediately beneficial in terms of advancing these techniques and offering the prospect of translation beyond their use as research tools. Of even greater value will be the public availability of the baseline model and a benchmark dataset, which can stimulate research across the community. 

## 5 Aims and objectives

We aim to develop generally applicable, freely available tools for using machine learning in X-ray computed tomography. Specifically we will develop:

- Open datasets for testing tomographic reconstruction and super-resolution approaches
- A benchmark baseline for tomographic reconstruction with neural networks
- A benchmark baseline for super-resolution enhancement of XRD-CT images using neural networks
- A documented, open-source repository for these tools and data

Our overarching aim is to accelerate machine learning application in XRD-CT by providing examples, baselines and test datasets. 

## 6 Methodology

### 6.1 Scientific methodology

The first part of the project focused on creating training libraries for the neural networks. To avoid bias, we have created libraries based on: (1) synthetic data containing random shapes using the well-known `scikit-image` Python package, (2) the DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges @ NTIRE (CVPR 2017 and CVPR 2018) and @ PIRM (ECCV 2018) and (3) images reconstructed from previously acquired micro-CT and XRD-CT experimental datasets. For the experimental tomographic datasets, the sinograms were first centered, scaled (i.e. assuming equal summed intensity per tomographic angle), background was subtracted (e.g. air scattering for the XRD-CT data) and then the images were reconstructing with the filtered back projection algorithm setting all negative values to zero. Specifically for the XRD-CT datasets, appropriate filters (i.e. trimmed mean filter) were applied to the raw 2D diffraction images during radial integration to avoid the formation of hotspots in the sinograms. These processed images are considered to be the ground truth for these libraries. Where needed, these images were rescaled to lower resolution using bilinear interpolation and artificial sinograms were created using the `astra toolbox` in Python. The performance of the reconstruction models is evaluated by comparing the reconstruction results of these sinograms with the ones obtained using the filtered back projection (i.e. reconstruction of synthetic sinograms). Similarly, for the super-resolution, the performance of the CNNs will be evaluated using the aforementioned three libraries.

### 6.2 AI methodology

We used a mixed architecture for CNN reconstruction. It starts with four 2D convolutional layers whose strides are equal to 2, followed by four fully connected layers. Then the 1D output from the last fully connected layer is transformed back to a 2D image and then sent to the next three 2D convolutional layers whose strides are equal to 1. For fully connected layers, there are 1000 nodes inside each of the first three of them, and the last fully connected layer has the size equal to the number of pixels of the reconstructed image. Besides, there is also one dropout layer after each fully connected layer to avoid overfitting. 

The accuracy and quality of the reconstructed images can be improved by increasing the number of nodes in the first three dense layers, but those numbers are significantly restricted by the computing resources. Increasing the number of nodes inside each layer can lead to a dramatic increase of the trainable weights, which makes the model harder to fit. Therefore, we added four more convolutional layers afterwards. The convolutional layers have significantly fewer weights than dense layers. They are used to take out the best-fitting features of the images and refine the reconstruction.

For the super-resolution, we are planning first to implement and evaluate the performance of the `EDSR`, `WDSR` and `SRGAN` neural networks using the aforementioned three libraries containing different types of image data. The next step will involve exploring their performance for upscaling micro-CT and for the first time, XRD-CT data. Finally, new architectures will be explored specifically for the XRD-CT data using a dual input of low resolution XRD-CT images and a high resolution micro-CT image acquired at the same position (i.e. XRD-CT and micro-CT corresponding at the same sample cross-section).

## 7 Interim results

### 7.1 Libraries

We have made a publicly available set of libraries for training and testing tomographic reconstruction and X-ray image super-resolution techniques. There is one library of 180,000 sinogram image pairs for reconstruction. As described above there are also several libraries of experimental micro-CT and XRD-CT data at different resolutions (for the same data), which can be used to train and test super-resolution methods. The links and details of the libraries can be found on the project documentation pages:  https://superres-tomo.readthedocs.io/en/latest/benchmark_data.html.

### 7.2 CNN tomographic reconstruction

For the purpose of this project, we have decided to use the currently latest stable version of Tensorflow (v.2.2.0) in python for the development and testing of the neural networks. We designed, implemented and evaluated a large number of CNN architectures, exploring also the impact of various hyperparameters (e.g. learning rate) and loss functions. From the various performance tests, we found the `cnn_reconstruct` discussed in the 6.2 section as one of the most promising ones, especially due to its ability for upscaling (i.e. it can handle relatively large images while maintaining a number of parameters in the order of 107 - 108). The learning rate was set to 0.00025 and the root mean squared was used as the loss function. We created a library using a combination of experimental XRD-CT datasets using catalyst particles consisting of 8,000 pairs of sinograms-images. Some examples are shown in Figure 2 where the original image and the ones obtained from the reconstruction using the filtered back projection algorithm and the cnn respectively are presented.

![img](https://lh3.googleusercontent.com/wTQCN8c8fjpY-3mmwWDFFHhv90Pc-cZCreuv2yIs7wX4k2ZBkdg7nAYNxcn9ERjnpWkP1uAzBc9uTSe2Oj2h4uSrPEZ5FCBah8RY3r3goGyMKFsIYkgE0fyxRC1ztqI3YLv0zkQ)*Figure 2 Performance of the reconstruction CNN and its comparison with the results obtained with the filtered back projection algorithm using a library containing XRD-CT sinograms-images.*

It can be seen here that the performance of the `cnn_reconstruct` is superior to the conventional filtered back projection algorithm as it can correctly reconstruct the shape and intensity of the catalyst particles while at the same time suppressing the background noise. However, when the same CNN was tested using the library containing random shapes (55,000 pairs of sinograms-images), the performance was worse. These results are presented in Figure 3. 

![img](https://lh4.googleusercontent.com/3C59kjTjkwXUP-uY9ntVLNizfEYS8S9OW2VmzQ0WJHPBb7n22ymoI4PWCOtl4rCTsrm9ExiwwTWsA3t-0NgL342HMyPgF33ds6n-SvO2oab7Oop75KYF106TqfQNivXxZ-yaU1w)*Figure 3 Performance of the reconstruction CNN and its comparison with the results obtained with the filtered back projection algorithm using a library containing synthetic sinograms-images of random shapes.*

It can be seen that the filtered back projection algorithm can retain the sharp edges of the shapes and their overall shape while the reconstruction CNN fails to do so. The problem here arises from the training data as the various shape images can vary significantly in content while there is a high degree of correlation between the XRD-CT images (i.e. the XRD-CT images present in each XRD-CT dataset). 

These results are very important as they illustrate the strong dependence of the reconstruction CNN on the training data and the difficulty in creating a reconstruction CNN that can handle very different data (i.e. images that are not well correlated with the training data). **This major issue is rarely discussed in literature and the current results from this project show that there should be more discussion on the impact and nature of training data used in supervised learning reconstruction CNNs.** As an example, in literature one can often encounter CNN's used in medical imaging (reconstruction, denoising etc) claimed to exhibit superior performance compared to conventional methods but the CNN's have been trained using only training libraries containing medical CT for a body part/organ. We are currently working on an unsupervised machine learning approach for image reconstruction using a Generative Adversarial Network (GAN) inspired by the recently published  `GANrec` which should eliminate this bias caused by the nature of the training data in supervised learning approaches.

### 7.3 MSDN for reconstruction/super-resolution

Initially in the proposed work plan we had intended to test the recently developed Mixed Scale Dense Networks (https://doi.org/10.1073/pnas.1715832114) for both image reconstruction and super-resolution. However when we further explored the architecture and its implementation, it has become apparent that the MSDN is restricted to inputs and outputs with exactly the same dimensions. This makes reconstruction impossible and limits the application for super-resolution. As a result we have decided to concentrate our efforts on Convolutional Neural Networks and Generative Adversarial Networks. 

## 8 Outputs

We have collected the outputs from the project in a repository, hosted on GitHub. We have also been building documentation to describe the outputs of the project, as well as an API, with extensive tutorials to allow others to use the tools and the data resulting from the project. The documentation can be found at: https://superres-tomo.readthedocs.io/en/latest/about.html and the code repository at https://github.com/keeeto/super-resolution-ml

### 8.1 Libraries

The data libraries are stored in `hdf5` format, to facilitate easy and efficient use in machine learning projects. The library locations and details are documented in the project docs at: https://superres-tomo.readthedocs.io/en/latest/benchmark_data.html

### 8.2 Networks

So far we have developed networks for image reconstruction, segmentation and denoising. We have packaged these together in a consistent fashion in our GitHub repository (https://github.com/keeeto/super-resolution-ml) to ensure that the code can be used by others. We have also written extensive tutorials to demonstrate how to train and apply the networks for these different tasks

* Reconstructions
    * With a CNN: https://superres-tomo.readthedocs.io/en/latest/tutorials.html#reconstruction-with-a-cnn
    * With a dense network: https://superres-tomo.readthedocs.io/en/latest/tutorials.html#reconstruction-with-a-dense-network
    * With an Automap network:
* Segmentation https://superres-tomo.readthedocs.io/en/latest/tutorials.html#segmentation-of-x-ray-images
* Denoising  https://superres-tomo.readthedocs.io/en/latest/tutorials.html#denoising-of-x-ray-images

## 9 Progress Summary

The project is advancing well.  We have completed the planned work in WP0, WP1 and WP2  by the end of the reporting period and made progress on some upcoming deliverables for WP3 (Table 1). Simulated and real micro CT and XRD-CT libraries have been constructed.  XRD-CT data can be simulated quickly and libraries have been built that contain large numbers of simulated diffraction patterning (with and without addition of noise) containing a  controlled spread of the embedded physico-chemical information. Details of exemplar datasets can be found in section 8 above. WP2 has been concerned with developing code for image reconstruction.  This was successfully achieved using CNN’s; the MSDN was abandoned for this task.  The work on the GAN’s has started and these approaches are giving good initial results.

| WP | Task / deliverable | Start | Days | End | Status | Comments |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Choice of AI strategies and algorithms | 01/05 | 15 | 11/05 | 100% | |
| 1 | Micro-CT and XRD-CT libraries | 01/05 | 40 | 15/06 | 100% | |
| 2 | CNN for image reconstruction | 15/05 | 40 | 29/06 | 100% | |
| 2 | MSDN for image reconstruction | 15/06 | 25 | 13/06 | 100% | approach failed |
| 2 | GAN for image reconstruction | 10/07 | 22 | 10/07 | 50% | |
| 3 | CNN for super-resolution | 15/08 | 50 | | 50% | |
| 3 | MSDN for super-resolution | 01/09 | 0 | | - | not viable route |
| 3 | GAN for super-resolution | 20/09 | 40 | | 0% | |
| 0 | Final report and paper(s) | 16/10 | 15 | | 0% | |

*NB The effort in WP3 has been updated to reflect the change in the plan due to the failure of the MSDN route ( highlighted in orange).*

## 10 Next steps

The next steps in the project are to complete the GAN development. The work on super-resolution in WP3 will begin shortly. The failure of the MSDN route will give us more time to work on the CNN and GAN approaches. Accordingly, we have modified our projected effort for these tasks.

We are going to produce further public datasets for image reconstruction.

We are also writing automated testing for the GitHub repository to ensure the stability of the code. We will continue to publish, document and test the new models and methods that we develop from the project. 

We hope to write the results of this project into some publications, potentially including publishing the software in the Journal of Open Source Software.