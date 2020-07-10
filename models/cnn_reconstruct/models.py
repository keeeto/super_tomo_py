from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Flatten, Reshape 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model

def reconstruct_cnn(ntr, npr):

    inputs = Input((ntr, npr, 1))

    convin1 = Conv2D(
        filters = 64, kernel_size = (3,3), strides = 2,
        padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(inputs)
    convin2 = Conv2D(
        filters = 64, kernel_size = (3,3), strides = 2, padding = 'same', activation = 'relu',
        kernel_initializer = 'he_normal')(convin1)
    convin3 = Conv2D(
        filters = 64, kernel_size = (3,3), strides = 2, padding = 'same', activation = 'relu',
        kernel_initializer = 'he_normal')(convin2)

    flatten = Flatten()(convin3)
    dense1 = Dense(ntr*ntr, activation = 'relu')(flatten)
    dense2 = Dense(ntr*ntr, activation = 'relu')(dense1)
    reshape = Reshape((ntr, ntr, 1))(dense2)


    conv1 = Conv2D(
        filters = 64, kernel_size = (7,7), strides = 2, padding = 'same', activation = 'relu',
        kernel_initializer = 'he_normal')(reshape)
    conv2 = Conv2D(
        filters = 64, kernel_size = (5,5), strides = 2, padding = 'same', activation = 'relu',
        kernel_initializer = 'he_normal')(conv1)
    conv3 = Conv2D(
        filters = 64, kernel_size = (3,3), strides = 2, padding = 'same', activation = 'relu',
        kernel_initializer = 'he_normal')(conv2)

    deconv1 = Conv2DTranspose(
        filters = 64, kernel_size = (7,7), strides = 2, padding = 'same', activation = 'relu',
        kernel_initializer = 'he_normal')(conv3)
    deconv2 = Conv2DTranspose(
        filters = 64, kernel_size = (5,5), strides = 2, padding = 'same', activation = 'relu',
        kernel_initializer = 'he_normal')(deconv1)
    deconv3 = Conv2DTranspose(
        filters = 64, kernel_size = (3,3), strides = 2, padding = 'same', activation = 'relu',
        kernel_initializer = 'he_normal')(deconv2)

    outputs = Conv2D(1, 3, padding='same')(deconv3)

    return Model(inputs, outputs)
    

