from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, Activation, Input, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Add
from keras.layers.pooling import AveragePooling2D
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.backend import tf as ktf

class LayerNorm(Layer):
    def __init__(self, epsilon=1e-7, beta_init='zero', gamma_init='one', **kwargs):
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        
        self.epsilon = epsilon
        self.uses_learning_phase = True
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = (input_shape[-1],)
        
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        
        self.trainable_weights = [self.gamma, self.beta]

    def call(self, x, mask=None):
        m, std = ktf.nn.moments(x, axes = [1, 2, 3], keep_dims=True)
        x_normed = (x - m) / (K.sqrt(std + self.epsilon))
        out = self.gamma * x_normed + self.beta
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {"epsilon": self.epsilon}
        base_config = super(LayerNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def resblock(x, kernel_size, resample, nfilters):
    assert resample in ["UP", "DOWN"]
   
    if resample == "DOWN":
        shortcut = AveragePooling2D(pool_size = (2, 2)) (x)
        shortcut = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                          padding = 'same', use_bias = True) (shortcut)        
        
        convpath = x
        convpath = LayerNorm() (x)
        convpath = Activation('relu') (convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                                 use_bias = False, padding='same')(convpath)
        convpath = AveragePooling2D(pool_size = (2, 2)) (convpath)
        convpath = LayerNorm() (convpath)
        convpath = Activation('relu') (convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                                 use_bias = True, padding='same') (convpath)        
        y = Add() ([shortcut, convpath])
    else:
        shortcut = UpSampling2D(size=(2, 2)) (x)        
        shortcut = Conv2D(nfilters, kernel_size, padding = 'same',
                          kernel_initializer='he_uniform', use_bias = True) (shortcut)
                
        convpath = BatchNormalization(axis=-1) (x)
        convpath = Activation('relu') (convpath)
        convpath = UpSampling2D(size=(2, 2))(convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform', 
                                 use_bias = False, padding='same')(convpath)        
        convpath = BatchNormalization(axis=-1) (convpath)
        convpath = Activation('relu') (convpath)
        convpath = Conv2D(nfilters, kernel_size, kernel_initializer='he_uniform',
                                 use_bias = True, padding='same') (convpath)
        
        y = Add() ([shortcut, convpath])
        
    return y                  

def make_generator():
    """Creates a generator model that takes a 512x4x4-dimensional noise vector as a "seed", and outputs images
    of size 64x64x3."""
    x = Input((128, ))
    y = Dense(512 * 4 * 4) (x)
    y = Reshape((4, 4, 512)) (y)
    
    y = resblock(y, (3, 3), 'UP', 512)
    y = resblock(y, (3, 3), 'UP', 256)
    y = resblock(y, (3, 3), 'UP', 128)
    y = resblock(y, (3, 3), 'UP', 64)
    
    y = BatchNormalization(axis=-1) (y)
    y = Activation('relu') (y)
    y = Conv2D(3, (3, 3), kernel_initializer='he_uniform', use_bias = False, 
                      padding='same', activation='tanh')(y)
    return Model(inputs=x, outputs=y) 


def make_generator_part():
    x = Input((128, ))
    y = Dense(512 * 4 * 4) (x)
    y = Reshape((4, 4, 512)) (y)
    
    y = resblock(y, (3, 3), 'UP', 512, 512)    
    
    return Model(inputs=x, outputs=y)

def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated."""
    x = Input((64, 64, 3))
    y = Conv2D(64, (3, 3), kernel_initializer='he_uniform',
                      use_bias = True, padding='same') (x)
    y = resblock(y, (3, 3), 'DOWN', 128)
    y = resblock(y, (3, 3), 'DOWN', 256)
    y = resblock(y, (3, 3), 'DOWN', 512)
    y = resblock(y, (3, 3), 'DOWN', 512)
    
    y = Flatten()(y)
    y = Dense(1, use_bias = False)(y)
    return Model(inputs=x, outputs=y)

