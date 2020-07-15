import numpy as np
import imageio
import time
import tensorflow as tf
from .models import compute_apply_gradients, compute_loss

def build_autoencoder_data(Xfiles, yfiles=None, input_data=(1, 1, 1)):
    '''
    A helper tool to build an input dataset from a list of files.

    Args:
        Xfiles: List of filenames
        yfiles: (optional) list of filenames, if not provided y=X
        input_data: the shape of the input data array (w, h, channels)
    Returns:
        X: a numpy array of matrices
        y: a numpy array of matrices
    '''
    X = np.zeros((len(Xfiles), input_data[0], input_data[1], input_data[2]))
    for i, im in enumerate(Xfiles):
        X[i] = np.expand_dims(imageio.imread(im), axis=2)
    X = np.float32(X)
    if not yfiles:
        labels = X
    else:
        labels = np.zeros((len(Xfiles), input_data[0], input_data[1], input_data[2]))
        for i, im in enumerate(yfiles):
            labels[i] = np.expand_dims(imageio.imread(im), axis=2)
        labels = np.float32(labels)
    return X, labels

def vae_inference(model, x, sigmoid=False):
   '''
   Run the VAE in inference mode

   Args:
       model: a tensorflow model, the VAE itself
       x: the example to run through the model
       sigmoid: boolean - apply sigmoid function on the final layer?
   '''
   mean, logvar = model.encode(x)
   z = model.reparameterize(mean, logvar)
   return model.decode(z, apply_sigmoid=sigmoid)

def vae_train(model, Xdata, ydata, Xval, yval, epochs, optimizer, sigmoid=False):
    '''
    Train the VAE model.    

    Args:
       model: a tensorflow model, the VAE itself
       xdata: training inputs
       ydata: training labels
       xval: validation inputs
       yval: validation labels
       epochs: number of epochs for training
       optimizer: tensorflow optimizer object to use
       sigmoid: boolean - apply sigmoid function on the final layer?
    '''

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for index, train_x in enumerate(Xdata):
            trx = np.expand_dims(train_x, axis=0)
            trl = np.expand_dims(ydata[index], axis=0)
            compute_apply_gradients(model, trx, trl, optimizer, sigmoid)
        end_time = time.time()
    
        if epoch % 5 == 0:
            loss = tf.keras.metrics.Mean()
            for index, test_x in enumerate(Xval):
                test_x = np.expand_dims(test_x, axis=0)
                test_y = np.expand_dims(yval[index], axis=0)
                loss(compute_loss(model, test_x, test_y, sigmoid))
            elbo = -loss.result()
            print('Epoch: {0:5d}, Test set ELBO: {1:10.4f}, '
            'time elapse for current epoch {2:8.3f}'.format(epoch,
                                                        elbo,
                                                       end_time - start_time))

