from small_res_architectures import LayerNorm

from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, Activation, Input, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Add, Concatenate
from keras.layers.pooling import AveragePooling2D
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.backend import tf as ktf
import numpy as np

from gan.layer_utils import LayerNorm, resblock


def output_conv(y):
    y = BatchNormalization(axis=-1) (y)
    y = Activation('relu') (y)
    y = Conv2D(3, (3, 3), kernel_initializer='he_uniform', padding='same', activation='tanh')(y)
    
    return y

def make_generator(noise, pose):
    y_pose_16_8 = resblock(pose, (3, 3), 'SAME', 256)
    y_pose_32_16 = resblock(y_pose_16_8, (3, 3), 'UP', 128)
    y_pose_64_32 = resblock(y_pose_32_16, (3, 3), 'UP', 64)
    y_pose_128_64 = resblock(y_pose_64_32, (3, 3), 'UP', 64)
    
    y = Dense(64 * 8 * 4) (noise)
    y = Reshape((8, 4, 64)) (y)
    
    y = resblock(y, (3, 3), 'UP', 64)
    y = resblock(y, (3, 3), 'SAME', 128)
    
    y = Concatenate(axis=-1) ([y, y_pose_16_8])
    
    y = resblock(y, (3, 3), 'UP', 64)
    y = resblock(y, (3, 3), 'SAME', 128)
    
    output_32_16 = output_conv(y)    
    y = Concatenate(axis=-1) ([output_32_16, y_pose_32_16])
    
    y = resblock(y, (3, 3), 'UP', 64)
    y = resblock(y, (3, 3), 'SAME', 128)
    
    output_64_32 = output_conv(y)
    y = Concatenate(axis=-1) ([output_64_32, y_pose_64_32])
    
    y = resblock(y, (3, 3), 'UP', 64)
    y = resblock(y, (3, 3), 'SAME', 64)
    y = Concatenate(axis=-1) ([y, y_pose_128_64])    
  
    y = resblock(y, (3, 3), 'SAME', 64)    
    output_128_64 = output_conv(y)
    
    return Model(inputs=[noise, pose], outputs=[output_128_64, output_64_32, output_32_16]) 


def make_discriminator(image_128_64, image_64_32, image_32_16):
    y_128_64 = Conv2D(16, (3, 3), kernel_initializer='he_uniform',
                      use_bias = True, padding='same') (image_128_64)
    
    y_64_32 = Conv2D(32, (3, 3), kernel_initializer='he_uniform',
                      use_bias = True, padding='same') (image_64_32)
    
    y_32_16 = Conv2D(64, (3, 3), kernel_initializer='he_uniform',
                      use_bias = True, padding='same') (image_32_16)
    
    y = resblock(y_128_64, (3, 3), 'DOWN', 32, LayerNorm)
    y = Concatenate(axis=-1) ([y, y_64_32])
    y = resblock(y, (3, 3), 'DOWN', 64, LayerNorm)
    y = Concatenate(axis=-1) ([y, y_32_16])
    y = resblock(y, (3, 3), 'DOWN', 256, LayerNorm)
    #y = Concatenate(axis=-1) ([y, y_pose_16_8])
    y = resblock(y, (3, 3), 'DOWN', 512, LayerNorm)
    
    y = Flatten()(y)
    y = Dense(1, use_bias = False)(y)
    return Model(inputs=[image_128_64, image_64_32, image_32_16], outputs=[y])
