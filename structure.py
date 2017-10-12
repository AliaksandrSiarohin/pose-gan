from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, Flatten, Activation, Input, Multiply, Concatenate, RepeatVector, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from gan.wgan_gp import WGAN_GP
from gan.dataset import FolderDataset
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
from skimage.transform import resize
from skimage.io import imread

from keras.models import load_model

def make_generator():
    noise = Input((64, ))
    y = Dense(256 * 2 * 1)(noise)
    y = Reshape((2, 1, 256)) (y)
    
    y = resblock(y, (3, 3), 'UP', 256)
    y = resblock(y, (3, 3), 'UP', 128)
    y = resblock(y, (3, 3), 'UP', 64)
    
    y = BatchNormalization(axis=-1) (y)
    y = Activation('relu') (y)
    y = Conv2D(18, (3, 3), kernel_initializer='he_uniform', padding='same', activation='tanh')(y)
    
    return Model(inputs=[noise], outputs=[y]) 


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated."""
    x = Input((16, 8, 18))
    y = Conv2D(32, (3, 3), kernel_initializer='he_uniform',
                      use_bias = True, padding='same') (x)
    
    y = resblock(y, (3, 3), 'DOWN', 64, LayerNorm)
    y = resblock(y, (3, 3), 'DOWN', 128, LayerNorm)
    y = resblock(y, (3, 3), 'DOWN', 256, LayerNorm)

    y = Flatten()(y)
    y = Dense(1, use_bias = False)(y)
    
    return Model(inputs=[x], outputs=[y])

class StuctureDataset(FolderDataset):
    def __init__(self, image_dir, image_size, batch_size, noise_size=(64, )):
        super(StuctureDataset, self).__init__(image_dir, batch_size, noise_size, image_size)
        self._pose_dir = 'tmp-pose-16-8-fasion'        
        self._pose_estimator = load_model('cao-hpe/pose_estimator.h5')
        self._precompute_pose_maps()
        names = [name.replace('.npy', '') for name in os.listdir(self._pose_dir)]
        self._image_names = np.array(names)
        print (self._image_names.shape)
        self._batches_before_shuffle = int(self._image_names.shape[0] // self._batch_size)
    
    def _precompute_pose_maps(self):
        print ("Precomputing pose_maps...")
        if not os.path.exists(self._pose_dir):
            os.makedirs(self._pose_dir)
        for name in tqdm(self._image_names):
            img = imread(os.path.join(self._input_dir, name))
            img = self._preprocess_image(img) / 2
            img = np.expand_dims(img, axis=0)
            pose = self._pose_estimator.predict(img)[1][..., :18][0]
            pose_size = pose.shape[:2]
            pose = resize(pose, self._image_size, preserve_range=True)
            pose = pose_utils.map_to_cord(pose)
            if len(np.where(pose != -1)[0]) > 10:
                pose = pose_utils.cords_to_map(pose, self._image_size)
                pose = resize(pose, pose_size, preserve_range=True)
                np.save(os.path.join(self._pose_dir, name), pose)
            
    def _load_discriminator_data(self, index):
        pose_batch = np.array([np.load(os.path.join(self._pose_dir, name + '.npy')) 
                                     for name in self._image_names[index]])
        return [pose_batch]

    def display(self, output_batch, input_batch = None, row=8, col=8):
        pose = output_batch
        pose = [resize(p, self._image_size, preserve_range=True) for p in pose]    
        imgs = np.array([pose_utils.draw_pose_from_map(p)[0] for p in pose])
        generated = super(FolderDataset, self).display(imgs, None, row, col)
        
        pose = self._load_discriminator_data(np.arange(row * col))[0]
        print (pose.shape)
        pose = [resize(p, self._image_size, preserve_range=True) for p in pose]
        imgs = np.array([pose_utils.draw_pose_from_map(p)[0] for p in pose])
        true = super(FolderDataset, self).display(imgs, None, row, col)

        return np.concatenate([generated, true], axis = 1)


def main():
    generator = make_generator()
    discriminator = make_discriminator()
    
    parser = parser_with_default_args()
    parser.add_argument("--annotations", default="cao-hpe/annotations.csv", help="File with pose annotations")
    args = parser.parse_args()

    dataset = StuctureDataset(args.input_folder, (128, 64), 64)
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


    
