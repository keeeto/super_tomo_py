****************
Models available
****************

This is a list and a brief description of the various models that are currently available 
in `superres-tomo`. More detailed information about each of the models can be found in 
their individual module documentaion. Additionally example use cases are provided in the
tutorials.


Automap
#######

Automap is a network architecture that was specifically developed for tomogrpahic reconstruction. [1]_ We have implemented
the architecture here. While we find that Automap does perform very well, we have alsofound that the dense connections mean
that the architecture memory requirements scale very badly and it cannot deal with many typical sinograms.

.. image:: figures/automap.png
   :width: 1100px
   :height: 512px
   :scale: 50 %
   :alt: alternate text
   :align: center

cnn_reconstruct
###############

This module contains a CNN model for tomographic image reconstruction. For more details on 
how to use this follow the tutorial on reconstruction with a CNN.

.. image:: figures/cnn-reconstruct.png
   :width: 1100px
   :height: 475px
   :scale: 50 %
   :alt: alternate text
   :align: center

dense_reconstruct
##################

This module contains a densely connected model for tomographic image reconstruction. For more details on 
how to use this follow the tutorial on reconstruction with a dense network.

.. image:: figures/dense.png
   :width: 1100px
   :height: 512px
   :scale: 50 %
   :alt: alternate text
   :align: center

AlexNet
#######

AlexNet is a specific instance of a CNN. It became famous in 2012, when it outperformed all of 
the competition in the ImageNet challenge. AlexNet contains eight layers; the first five are 
convolutional layers, some of them followed by max-pooling layers, and the last three are fully 
connected layers. [2]_ It uses the ReLU activation function.

U-Net
#####

The U-net architecture is particularly popular in image segmentation tasks. The model consists
of convolution layers mirrored by deconvolution layers, squeezing down and then reconstructing 
an image of the same size as the original. Doing this a U-Net can convert pixels in the original
image into label values and segment the image. U-nets were initially developed for biomedical 
image segmentation. [3]_

.. image:: figures/unet.png
   :width: 1100px
   :height: 512px
   :scale: 50 %
   :alt: alternate text
   :align: center


Autoencoder
###########

An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. [4]_ The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise”. For this reason autoencoders can be used for denoising images.



References
##########

.. [1] Bo Zhu, Jeremiah Z. Liu, Bruce R. Rosen, Matthew S. Rosen "Image reconstruction by 
   domain transform manifold learning" https://arxiv.org/pdf/1704.08841.pdf

.. [2] Krizhevsky, Alex; Sutskever, Ilya; Hinton, Geoffrey E. (2017-05-24). "ImageNet 
    classification with deep convolutional neural networks" (PDF). Communications of the ACM. 60 
    (6): 84–90

.. [3] Ronneberger, Olaf; Fischer, Philipp; Brox, Thomas (2015). "U-Net: Convolutional 
    Networks for Biomedical Image Segmentation". arXiv:1505.04597

.. [4] Vincent, Pascal; Larochelle, Hugo (2010). "Stacked Denoising Autoencoders: Learning 
   Useful Representations in a Deep Network with a Local Denoising Criterion". Journal 
   of Machine Learning Research. 11: 3371–3408

