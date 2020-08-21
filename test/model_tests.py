import os
import pytest
import numpy as np
import sys

sys.path.append('/home/mts87985/ml-tomo/')
from super_tomo_py.models.alex_net.models import alex_net_classify

# Random seed to ensure that tests are repeatable
RANDOM_SEED = 23
np.random.seed(RANDOM_SEED)

def test_alex():
    width = np.random.randint(300, 400)
    height = np.random.randint(300, 400)
    channels = np.random.randint(1, 4)
    image = np.random.randint(0, 255, size=(width, height, channels))
    image =  np.expand_dims(image, axis=0)
    model = alex_net_classify(width, height, channels)
    result = model.predict(image)
    assert result.shape == (1, 2)
    assert 0 <= result[0][0] <= 1
    assert 0 <= result[0][1] <= 1

def test_automap():
    from tensorflow.keras.optimizers import Adam
    from super_tomo_py.models.automap.models import automap

    width = np.random.randint(100, 200)
    height = np.random.randint(100, 200)
    channels = 1
    image = np.random.randint(0, 255, size=(width, height, channels))
    sinos =  np.expand_dims(image, axis=0)
    model_rec = automap(sinos.shape[1], sinos.shape[2])
    pred = model_rec.predict(sinos)

    assert pred.shape == (1, width, width)

def test_cnn_reconstruct():
    from super_tomo_py.models.cnn_reconstruct.models import reconstruct_cnn
    width = np.random.randint(100, 200)
    height = np.random.randint(100, 200)
    channels = 1
    image = np.random.randint(0, 255, size=(width, height, channels))
    sinos =  np.expand_dims(image, axis=0)
    model = reconstruct_cnn(sinos.shape[1], sinos.shape[2])
    pred = model.predict(sinos)

    assert pred.shape == (1, width, width)

def test_dense_reconstruct():
    from super_tomo_py.models.dense_reconstruct.models import dense_reconstruct
    width = np.random.randint(100, 200)
    height = np.random.randint(100, 200)
    channels = 1
    image = np.random.randint(0, 255, size=(width, height, channels))
    sinos =  np.expand_dims(image, axis=0)
    model = dense_reconstruct(sinos.shape[1], sinos.shape[2])
    pred = model.predict(sinos)

    assert pred.shape == (1, width, width)

def test_unet_reg():
    from super_tomo_py.models.u_net.models import unet

    width = np.random.randint(8, 18)*16
    height = np.random.randint(8, 18)*16
    channels = np.random.randint(1, 4)
    image = np.random.randint(0, 255, size=(width, height, channels))
    image =  np.expand_dims(image, axis=0)
    model = unet((width, height, channels))
    pred = model.predict(image)

    assert pred.shape == (1, width, height, 1)

def test_unet_4():
    from super_tomo_py.models.u_net.models import unet_4layer

    width = np.random.randint(8, 18)*8
    height = np.random.randint(8, 18)*8
    channels = np.random.randint(1, 4)
    image = np.random.randint(0, 255, size=(width, height, channels))
    image =  np.expand_dims(image, axis=0)
    model = unet_4layer((width, height, channels))
    pred = model.predict(image)

    assert pred.shape == (1, width, height, 1)

def test_unet_3():
    from super_tomo_py.models.u_net.models import unet_3layer

    width = np.random.randint(8, 18)*4
    height = np.random.randint(8, 18)*4
    channels = np.random.randint(1, 4)
    image = np.random.randint(0, 255, size=(width, height, channels))
    image =  np.expand_dims(image, axis=0)
    model = unet_3layer((width, height, channels))
    pred = model.predict(image)

    assert pred.shape == (1, width, height, 1)
