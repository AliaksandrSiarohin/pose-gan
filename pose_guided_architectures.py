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

def make_generator(noise, pose):
    y_pose_16_8 = resblock(pose, (3, 3), 'SAME', 256)
    y_pose_32_16 = resblock(y_pose_16_8, (3, 3), 'UP', 128)
    y_pose_64_32 = resblock(y_pose_32_16, (3, 3), 'UP', 64)
    y_pose_128_64 = resblock(y_pose_64_32, (3, 3), 'UP', 64)
    
    y = Dense(256 * 8 * 4) (noise)
    y = Reshape((8, 4, 256)) (y)
    
    y = resblock(y, (3, 3), 'UP', 256)
    y = resblock(y, (3, 3), 'SAME', 256)
    y = Concatenate(axis=-1) ([y, y_pose_16_8])
    y = resblock(y, (3, 3), 'UP', 128)
    y = resblock(y, (3, 3), 'SAME', 128)
    y = Concatenate(axis=-1) ([y, y_pose_32_16])
    y = resblock(y, (3, 3), 'UP', 64)
    y = resblock(y, (3, 3), 'SAME', 64)
    y = Concatenate(axis=-1) ([y, y_pose_64_32])
    y = resblock(y, (3, 3), 'UP', 64)
    y = resblock(y, (3, 3), 'SAME', 64)
    y = Concatenate(axis=-1) ([y, y_pose_128_64])    
  
    y = resblock(y, (3, 3), 'SAME', 64)
    
    y = BatchNormalization(axis=-1) (y)
    y = Activation('relu') (y)
    y = Conv2D(3, (3, 3), kernel_initializer='he_uniform', use_bias = False, 
                      padding='same', activation='tanh')(y)
    return Model(inputs=[noise, pose], outputs=[y]) 


def make_discriminator(image):
    #y_pose_16_8 = resblock(pose, (3, 3), 'SAME', 256, LayerNorm)
    #y_pose_32_16 = resblock(y_pose_16_8, (3, 3), 'UP', 128, LayerNorm)
    #y_pose_64_32 = resblock(y_pose_32_16, (3, 3), 'UP', 64, LayerNorm)
    
    y = Conv2D(32, (3, 3), kernel_initializer='he_uniform',
                      use_bias = True, padding='same') (image)
    
    y = resblock(y, (3, 3), 'DOWN', 64, LayerNorm)
    #y = Concatenate(axis=-1) ([y, y_pose_64_32])
    y = resblock(y, (3, 3), 'DOWN', 128, LayerNorm)
    #y = Concatenate(axis=-1) ([y, y_pose_32_16])
    y = resblock(y, (3, 3), 'DOWN', 256, LayerNorm)
    #y = Concatenate(axis=-1) ([y, y_pose_16_8])
    y = resblock(y, (3, 3), 'DOWN', 512, LayerNorm)
    
    y = Flatten()(y)
    y = Dense(1, use_bias = False)(y)
    return Model(inputs=[image], outputs=[y])
