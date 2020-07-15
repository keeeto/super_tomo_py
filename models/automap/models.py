from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Flatten, Reshape 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model

def automap(npr, ntr):
   '''
   Builds the Automap model, from Nature volume 555, pages487–492(2018)

   Args:
       ntr: (int) Number of translation steps
       npr: (int) Number of tomographic angles
   Returns:
       A tensorflow model
   '''

   input_tensor = Input((npr, ntr, 1))

   hidden_1 = Flatten()(input_tensor)
   hidden_2 = Dense(256, activation = 'tanh')(hidden_1)
   hidden_3 = Dense(256, activation = 'tanh')(hidden_2)
   hidden_4 = Dense(256, activation = 'tanh')(hidden_3)
   hidden_5 = Dense(ntr*ntr, activation = 'tanh')(hidden_4)

   FC_M = Reshape((ntr, ntr, 1))(hidden_5)

   hidden_6 = Conv2D(
       filters = 64,
       kernel_size = 5,
       strides=(1, 1),
       padding = 'same',
       activation = 'relu')(FC_M)

   hidden_7 = Conv2D(
       filters = 64,
       kernel_size = 5,
       strides=(1, 1),
       padding = 'same',
       activation = 'relu')(hidden_6)

   hidden_8 = Conv2DTranspose(

       filters = 1,
       kernel_size = 7,
       strides=(1, 1),
       padding = 'same',
       data_format = 'channels_last',
       activation = 'relu')(hidden_7)
   
   hidden_9 = Flatten()(hidden_8)

   hidden_10 = Dense(ntr*ntr, activation = 'linear')(hidden_9)

   output_tensor = Reshape((ntr, ntr))(hidden_10)

   return Model(input_tensor, output_tensor)
