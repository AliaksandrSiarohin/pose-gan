from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, Flatten, Activation, Input, Multiply, Concatenate, RepeatVector, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from gan.wgan_gp import WGAN_GP
from gan.dataset import ArrayDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer
from gan.layer_utils import LayerNorm, resblock

import numpy as np
import pandas as pd
import pose_utils

from keras.engine.topology import Layer
from keras import initializers
from keras.backend import tf as ktf
import keras.backend as K

import os
from tqdm import tqdm

def make_generator():
    """Creates a generator model that takes a 128-dimensional noise vector as a "seed", and outputs images
    of size 128x64x3."""
    x = Input((64, ))
    y = Dense(128, use_bias=False)(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Dense(256, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Dense(128, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)    
    
    y = Dense(5 * 4, use_bias=True)(y)
    y = Activation('tanh')(y)

    return Model(inputs=[x], outputs=[y])


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated."""
    x = Input((5 * 4,))

    y = Dense(64, use_bias=True)(x)
    y = LeakyReLU()(y)

    y = Dense(128, use_bias=True)(y)
    y = LeakyReLU()(y)

    y = Dense(64, use_bias=True)(y)
    y = LeakyReLU()(y)

    y = Dense(1, use_bias=True)(y)
    return Model(inputs=[x], outputs=[y])


class StuctureDataset(ArrayDataset):
    def __init__(self, annotations, img_size, batch_size, noise_size=(64, )):
        self._img_size = img_size
        X = []
        for name in tqdm(os.listdir('../market-dataset/bounding_box_train/')):
            ano = np.load('../edge_boxes_with_python/market-bbox-propose/' + name + '.npy')
            X.append(ano[0:5, 0:4])
        X = self._preprocess_array(np.array(X))
        super(StuctureDataset, self).__init__(X, batch_size, noise_size)
    
    def _preprocess_array(self, X):
        X[:,:,0::2] /= self._img_size[0]
        X[:,:,1::2] /= self._img_size[1]        
        X = 2 * (X - 0.5)
        return X.reshape((X.shape[0], -1))

    def _deprocess_array(self, X):
        X = X / 2 + 0.5
        X = X.reshape((X.shape[0], 5, 4))
        mask = X < 0
        X[...,0::2] *= self._img_size[0] - 0.1
        X[...,1::2] *= self._img_size[1] - 0.1
        X[mask] = -1
        return X.astype(np.int)

    def display(self, output_batch, input_batch = None, row=8, col=8):
        def to_image(batch):            
            output_batch = self._deprocess_array(batch)       
            imgs = [pose_utils.bbox_to_map(cord, self._img_size) for cord in output_batch]
            imgs = np.array([img / np.max(img) for img in imgs])
            imgs = imgs[..., np.newaxis]            
            res = super(StuctureDataset, self).display(imgs, None, row, col)
            return np.squeeze(res, axis=-1)        
        return np.concatenate([to_image(output_batch), to_image(self._X[:row * col])], axis = 1)


def main():
    generator = make_generator()
    discriminator = make_discriminator()
    
    parser = parser_with_default_args()
    parser.add_argument("--annotations", default="cao-hpe/annotations.csv", help="File with pose annotations")
    args = parser.parse_args()

    dataset = StuctureDataset(args.annotations, (128, 64), 64)
    gan = WGAN_GP(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()

if __name__ == "__main__":
    main()
    # import pylab as plt
    # dataset = StuctureDataset("cao-hpe/annotations.csv", (128, 64), 64)
    # pose = dataset._X[0:1]
    #
    # pose = dataset._deprocess_array(pose)
    # print (pose)
    #
    # plt.imsave('1.png', pose_utils.draw_pose_from_cords(pose[0], (128, 64))[0])
    #plt.show()
    #plt.savefig('1.png')

    #
    # x = Input((18, 2))
    # y = MissingEmbeding()(x)
    # a = Model(inputs=[x], outputs=[y])
    # print (a.predict(np.expand_dims(pose, 0)))


    
