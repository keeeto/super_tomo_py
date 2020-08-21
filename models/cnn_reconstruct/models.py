from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Flatten, Reshape 
from tensorflow.keras.layers import Conv2DTranspose, Dropout
from tensorflow.keras.models import Model

def reconstruct_cnn(ntr, npr):
    '''
    Builds a cnn reconstruction model

    Args:
       ntr: (int) Number of translation steps
       npr: (int) Number of tomographic angles
    Returns:
       A tensorflow model
    '''

    inputs = Input(shape = (ntr,npr,1))
    conv = Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = 'same', activation = 'relu')(inputs)
    conv = Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = 'same', activation = 'relu')(conv)
    conv = Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = 'same', activation = 'relu')(conv)
    conv = Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = 'same', activation = 'relu')(conv)
    flat = Flatten()(conv)
    dense = Dense(1000, activation = 'relu')(flat)
    drop = Dropout(0.1)(flat)
    dense = Dense(1000, activation = 'relu')(drop)
    drop = Dropout(0.1)(flat)
    dense = Dense(1000, activation = 'relu')(drop)
    drop = Dropout(0.1)(flat)
    dense = Dense(1000, activation = 'relu')(drop)
    drop = Dropout(0.1)(flat)
    dense = Dense(ntr*ntr, activation = 'linear')(drop)
    reshape = Reshape((ntr,ntr,1))(dense)
    conv = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(reshape)
    conv = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv)
    conv = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(conv)
    outputs = Conv2D(1, 3, padding='same')(conv)
    outputs = Reshape((ntr, ntr))(outputs)

    return Model(inputs, outputs)
    

