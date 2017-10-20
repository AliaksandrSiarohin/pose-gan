from keras.models import Model
from keras.layers import Dense, Reshape, Flatten, Activation, Input
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers.normalization import InstanceNormalization

from gan.wgan_gp import WGAN_GP
from gan.dataset import FolderDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer
from gan.layer_utils import resblock

import numpy as np

def make_generator():
    """Creates a generator model that takes a 128-dimensional noise vector as a "seed", and outputs images
    of size 128x64x3."""
    x = Input((128, ))
    y = Dense(512 * 8 * 4)(x)
    y = Reshape((8, 4, 512))(y)
    
    y = resblock(y, (3, 3), 'UP', 512)
    y = resblock(y, (3, 3), 'UP', 256)
    y = resblock(y, (3, 3), 'UP', 128)
    y = resblock(y, (3, 3), 'UP', 64)
    
    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    y = Conv2D(3, (3, 3), kernel_initializer='he_uniform', use_bias = False, 
                      padding='same', activation='tanh')(y)
    return Model(inputs=x, outputs=y) 


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated."""
    x = Input((128, 64, 3))
    y = Conv2D(64, (3, 3), kernel_initializer='he_uniform',
                      use_bias=True, padding='same')(x)
    y = resblock(y, (3, 3), 'DOWN', 128, InstanceNormalization)
    y = resblock(y, (3, 3), 'DOWN', 256, InstanceNormalization)
    y = resblock(y, (3, 3), 'DOWN', 512, InstanceNormalization)
    y = resblock(y, (3, 3), 'DOWN', 512, InstanceNormalization)
    
    y = Flatten()(y)
    y = Dense(1, use_bias = False)(y)
    return Model(inputs=x, outputs=y)

def main():
    generator = make_generator()
    discriminator = make_discriminator()
    
    args = parser_with_default_args().parse_args()
    dataset = FolderDataset(args.input_folder, args.batch_size, (128, ), (128, 64))
    gan = WGAN_GP(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()

if __name__ == "__main__":
    main()
    
