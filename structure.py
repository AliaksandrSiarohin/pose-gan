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

    y = Dense(512, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    
    y = Dense(256, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Dense(128, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Dense(64, use_bias=False)(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)    
    
    y = Dense(18 * 2, use_bias=True)(y)
    y = Activation('tanh')(y)
    
   
    mask = Dense(18 * 2, use_bias=True)(y)
    mask = Activation('sigmoid')(mask)    
 
    return Model(inputs=[x], outputs=[y, mask])


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated."""
    x = Input((18 * 2,))
    mask = Input((18 * 2,))
    
    y = Multiply()([x, mask])
    y = Concatenate(axis=-1) ([y, mask])

    y = Dense(64, use_bias=True)(y)
    y = LeakyReLU()(y)

    y = Dense(128, use_bias=True)(y)
    y = LeakyReLU()(y)

    y = Dense(256, use_bias=True)(y)
    y = LeakyReLU()(y)

    y = Dense(128, use_bias=True)(y)
    y = LeakyReLU()(y)

    y = Dense(64, use_bias=True)(y)
    y = LeakyReLU()(y)

    y = Dense(32, use_bias=True)(y)
    y = LeakyReLU()(y)
    
#     y_mask = Dense(64, use_bias=True)(mask)
#     y_mask = LeakyReLU()(y_mask)
#     y_mask = Dense(32, use_bias=True)(y_mask)
#     y_mask = LeakyReLU()(y_mask)
#     y_mask = Dense(16, use_bias=True)(y_mask)
#     y_mask = LeakyReLU()(y_mask)
    
#     y = Concatenate(axis=-1)([y, y_mask])
    
    y = Dense(1, use_bias=True)(y)
    return Model(inputs=[x, mask], outputs=[y])


class StuctureDataset(ArrayDataset):
    def __init__(self, annotations, img_size, batch_size, noise_size=(64, )):
        df = pd.read_csv(annotations, sep=':')
        self._img_size = img_size
        X = []
        for index, row in df.iterrows():
            X.append(pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x']))
        X = np.array(X, dtype='float32')
        self._mask = self._get_mask(X)
        X = self._preprocess_array(X)
        self._seed = 0
        super(StuctureDataset, self).__init__(X, batch_size, noise_size)
    
    def _get_mask(self, X):
        return (X != -1).astype('float32')
    
    def _load_discriminator_data(self, index):
        mask = self._mask[index]
        mask = mask.reshape((mask.shape[0], -1))
        return [self._X[index], mask]
    
    def _shuffle_discriminator_data(self):
        self._seed += 1
        np.random.seed(self._seed)
        np.random.shuffle(self._X)
        np.random.seed(self._seed)
        np.random.shuffle(self._mask)
    
    def _preprocess_array(self, X):
        X[:,:,0] /= self._img_size[0]
        X[:,:,1] /= self._img_size[1]
        X = 2 * (X - 0.5)
        return X.reshape((X.shape[0], -1))

    def _deprocess_array(self, X):
        X = X / 2 + 0.5
        X = X.reshape((X.shape[0], 18, 2))
        mask = X < 0
        X[...,0] *= self._img_size[0] - 0.1
        X[...,1] *= self._img_size[1] - 0.1
        X[mask] = -1
        return X.astype(np.int)

    def display(self, output_batch, input_batch = None, row=64, col=1):
        pose, mask = output_batch[0], output_batch[1]
        
        output_batch = self._deprocess_array(pose)
        output_batch[mask.reshape((-1, 18, 2)) < 0.5] = -1        
        imgs = np.array([pose_utils.draw_pose_from_cords(cord, self._img_size)[0] for cord in output_batch])
        generatred_masked = super(StuctureDataset, self).display(imgs, None, row, col)
        
        output_batch = self._deprocess_array(pose)       
        imgs = np.array([pose_utils.draw_pose_from_cords(cord, self._img_size)[0] for cord in output_batch])
        generated_not_masked = super(StuctureDataset, self).display(imgs, None, row, col)

        output_batch = self._deprocess_array(self._X[:64])
        imgs = np.array([pose_utils.draw_pose_from_cords(cord, self._img_size)[0] for cord in output_batch])
        true = super(StuctureDataset, self).display(imgs, None, row, col)

        return np.concatenate([generated_not_masked, generatred_masked, true], axis = 1)


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


    
