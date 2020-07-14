**************
Loss functions
**************

`superres-tomo` comes with a series of loss functions not found in the ml packages that it is built
on. These are primarily useful in cases where there is class imbalance in the training/test set. 


Weighted cross entropy
######################

The weighted cross entropy (WCE) is an extension of the standard cross entropy. WCE can be used where
there is a large class imbalance (excess of a particular label). In WCE all positive examples get 
weighted by a coefficient, which can be set in inverse proportion to the amount of a given label in 
the training set.

WCE is defined as

.. math::

   WCE(p, \hat{p}) = -(\beta p \log(\hat{p}) + (1 - p)\log(1 - \hat{p}))

To bias against false positives set :math:`\beta < 1` to bias against false positives set :math:`\beta < 1`

Balanced cross entropy
######################

Balanced cross entropy (BCE) is similar to WCE, but it also biases negatives as well as poitives.

.. math::

   BCE(p, \hat{p}) = -(\beta p \log(\hat{p}) + (1 - \beta)(1 - p)\log(1 - \hat{p}))

Dice Loss
#########

The Dice loss is a loss function that is particularly useful if boundary detection is important in your image
analysis. The dice loss is defined as 

.. math::

   DL(p, \hat{p}) = 1 - \frac{2p\hat{p} + 1}{p + \hat{p} +1}

