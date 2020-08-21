"""
Created on July 09 2020
@author: Keith Butler
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import initializers
import keras.backend as K


def alex_net_classify(img_height, img_width, n_channels=3, n_classes=2, 
                      str_init='initializers.glorot_normal(seed=None)'):
    """
            A basic set up of the AlexNet CNN architecture https://doi.org/10.1145%2F3065386
	    for classification
	    Args:
	        img_height: the height of the image in pixels (int)
	        img_width: the width of the image in pixels (int)
		n_classes: number of clsses
		str_init: the initialiser to use for the weights of the netowrk 
		    (str, see https://keras.io/initializers/, 
		    default 'initializers.glorot_normal(seed=None)')

	    Returns:
	        model: a Keras model
	        train_generator: a generator for the training data
	        test_generator: a generator for the testing data
    """
    init = eval(str_init)
    #K.set_image_dim_ordering('tf')
##### BUILD THE NETWORK

# (3) Create a sequential model
    model = Sequential()

# 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(img_height, img_width, n_channels),
                     kernel_size=(15, 5), strides=(5, 2), padding='valid',
                     kernel_initializer=init))

    model.add(Activation('relu'))
# Pooling 
    model.add(MaxPooling2D(pool_size=(4,4), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

# 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid' 
          , kernel_initializer=init))
    model.add(Activation('relu'))
# Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
    #model.add(BatchNormalization())

# 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'
                    , kernel_initializer=init))
    model.add(Activation('relu'))
# Batch Normalisation
    #model.add(BatchNormalization())

# 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'
              , kernel_initializer=init))
    model.add(Activation('relu'))
# Batch Normalisation
    model.add(BatchNormalization())

# 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'
              , kernel_initializer=init))
    model.add(Activation('relu'))
# Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
# Batch Normalisation
    #model.add(BatchNormalization())

# Passing it to a dense layer
    model.add(Flatten())
# 1st Dense Layer
    model.add(Dense(1096, input_shape=(img_width*img_height*3,), kernel_initializer=init))
    model.add(Activation('relu'))
# Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
# Batch Normalisation
    #model.add(BatchNormalization())

# 2nd Dense Layer
    model.add(Dense(1096, kernel_initializer=init))
    model.add(Activation('relu'))
# Add Dropout
    #model.add(Dropout(0.4))
# Batch Normalisation
    model.add(BatchNormalization())

# 3rd Dense Layer
    model.add(Dense(1000, kernel_initializer=init))
    model.add(Activation('relu'))
# Add Dropout
    model.add(Dropout(0.4))
# Batch Normalisation
    model.add(BatchNormalization())

# Output Layer
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))


    print(model.summary())

    return model

def alex_net_regress(img_height, img_width, n_channels=3, n_classes=2, 
                      str_init='initializers.glorot_normal(seed=None)'):
    """
            A basic set up of the AlexNet CNN architecture https://doi.org/10.1145%2F3065386
	    for classification
	    Args:
	        img_height: the height of the image in pixels (int)
	        img_width: the width of the image in pixels (int)
		n_classes: number of clsses
		str_init: the initialiser to use for the weights of the netowrk 
		    (str, see https://keras.io/initializers/, 
		    default 'initializers.glorot_normal(seed=None)')

	    Returns:
	        model: a Keras model
	        train_generator: a generator for the training data
	        test_generator: a generator for the testing data
    """
    init = eval(str_init)
    #K.set_image_dim_ordering('tf')
##### BUILD THE NETWORK

# (3) Create a sequential model
    model = Sequential()

# 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(img_height, img_width, n_channels),
                     kernel_size=(15, 5), strides=(5, 2), padding='valid',
                     kernel_initializer=init))

    model.add(Activation('relu'))
# Pooling 
    model.add(MaxPooling2D(pool_size=(4,4), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

# 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid' 
          , kernel_initializer=init))
    model.add(Activation('relu'))
# Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
    #model.add(BatchNormalization())

# 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'
                    , kernel_initializer=init))
    model.add(Activation('relu'))
# Batch Normalisation
    #model.add(BatchNormalization())

# 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'
              , kernel_initializer=init))
    model.add(Activation('relu'))
# Batch Normalisation
    model.add(BatchNormalization())

# 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'
              , kernel_initializer=init))
    model.add(Activation('relu'))
# Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
# Batch Normalisation
    #model.add(BatchNormalization())

# Passing it to a dense layer
    model.add(Flatten())
# 1st Dense Layer
    model.add(Dense(1096, input_shape=(img_width*img_height*3,), kernel_initializer=init))
    model.add(Activation('relu'))
# Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
# Batch Normalisation
    #model.add(BatchNormalization())

# 2nd Dense Layer
    model.add(Dense(1096, kernel_initializer=init))
    model.add(Activation('relu'))
# Add Dropout
    #model.add(Dropout(0.4))
# Batch Normalisation
    model.add(BatchNormalization())

# 3rd Dense Layer
    model.add(Dense(1000, kernel_initializer=init))
    model.add(Activation('relu'))
# Add Dropout
    model.add(Dropout(0.4))
# Batch Normalisation
    model.add(BatchNormalization())

# Output Layer
    model.add(Dense(n_classes))
    model.add(Activation('linear'))


    print(model.summary())

    return model
