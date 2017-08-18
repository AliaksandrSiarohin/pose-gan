from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.activations import ReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Add
from keras.layers.pooling import AveragePooling2D

def resblock(x, kernel_size, resample, nfilters):
    assert resample in ["UP", "DOWN"]
   
    if resample == "DOWN":
        shortcut = Convolution2D(nfilters, kernel_size, border_mode = 'same') (x)
        shortcut = AveragePooling2D(pool_size = (2, 2)) (shortcut)
        
        convpath = BatchNormalization(axis=-1) (x)
        convpath = ReLU() (convpath)
        convpath = Convolution2D(nfilters, kernel_size, kernel_initializer='he_normal',
                                 use_bias = False, border_mode='same')(convpath)
        convpath = AveragePooling2D(pool_size = (2, 2)) (convpath)
        convpath = BatchNormalization(axis=-1) (convpath)
        convpath = ReLU() (convpath)
        convpath = Convolution2D(nfilters, kernel_size, kernel_initializer='he_normal',
                                 use_bias = False, border_mode='same') (convpath)
        
        y = Add() ([shortcut, convpath])
    else:
        shortcut = Convolution2D(nfilters, kernel_size, border_mode = 'same') (x)
        shortcut = UpSampling2D(size=(2, 2))
        
        convpath = BatchNormalization(axis=-1) (x)
        convpath = ReLU() (convpath)
        convpath = Convolution2D(nfilters, kernel_size, kernel_initializer='he_normal', 
                                 strides=[2, 2], use_bias = False, border_mode='same')(convpath)
        convpath = UpSampling2D(size=(2, 2))(convpath)
        convpath = BatchNormalization(axis=-1) (convpath)
        convpath = ReLU() (convpath)
        convpath = Convolution2D(nfilters, kernel_size, kernel_initializer='he_normal',
                                 use_bias = False, border_mode='same') (convpath)
        
        y = Add() ([shortcut, convpath])
        
    return Model(inputs=x, outputs=y) 
        
                  

def make_generator():
    """Creates a generator model that takes a 64x4x4-dimensional noise vector as a "seed", and outputs images
    of size 28x28x1."""
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(LeakyReLU())
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    bn_axis = -1
    model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    # Because we normalized training inputs to lie in the range [-1, 1],
    # the tanh function should be used for the output of the generator to ensure its output
    # also lies in this range.
    model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
    Instead, the output should be as large and negative as possible for generated inputs and as large and positive
    as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
    model = Sequential()
    model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model
