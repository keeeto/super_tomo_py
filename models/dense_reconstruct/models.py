from tensorflow.keras.layers import Dense, Conv2D, Input
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.models import Model

def dense_reconstruct(ntr, npr):
    '''
    Builds a dense reconstruction model
    Args:
       ntr: (int) Number of translation steps
       npr: (int) Number of tomographic angles
    Returns:
       A tensorflow model
    '''
    inputs = Input((ntr, npr, 1))
    flat1 = Flatten(input_shape=(ntr, npr))(inputs)
    dense1 = Dense(10*ntr, activation = 'relu')(flat1)
    dense1 = Dense(10*ntr, activation = 'relu')(dense1)
    dense1 = Dense(10*ntr, activation = 'relu')(dense1)
    dense1 = Dense(10*ntr, activation = 'relu')(dense1)
    dense1 = Dense(ntr*ntr, activation = 'relu')(dense1)
    reshape = Reshape((ntr,ntr,1))(dense1)
    conv1 = Conv2D(filters = 64, kernel_size = (3,3), 
                   strides = 1, padding = 'same', activation = 'relu')(reshape)
    conv1 = Conv2D(filters = 64, kernel_size = (3,3), 
                   strides = 1, padding = 'same', activation = 'relu')(conv1)
    conv1 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1,
                   padding = 'same', activation = 'relu')(conv1)
    out1 = Conv2D(1, 3, padding='same')(conv1)

    return Model(inputs = inputs, outputs = out1)
    
