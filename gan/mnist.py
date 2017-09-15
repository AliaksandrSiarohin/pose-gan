from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from wgan_gp import WGAN_GP
from dataset import ArrayDataset
from cmd import parser_with_default_args
from train import Trainer

import numpy as np

def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed", and outputs images
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

class MNISTDataset(ArrayDataset):
    def __init__(self, batch_size, noise_size = (100, )):
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate((X_train, X_test), axis=0)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        X = (X.astype(np.float32) - 127.5) / 127.5
        super(MNISTDataset, self).__init__(X, batch_size, noise_size)
        
    def display(self, output_batch, input_batch = None, row=8, col=8):
        batch = output_batch
        image = super(MNISTDataset, self).display(batch, row, col)
        image = (image * 127.5) + 127.5
        image = np.squeeze(np.round(image).astype(np.uint8))
        return image

def main():
    generator = make_generator()
    discriminator = make_discriminator()
    
    args = parser_with_default_args().parse_args()
    dataset = MNISTDataset(args.batch_size)
    gan = WGAN_GP('output/checkpoints/epoch_019_generator.png',
                  'output/checkpoints/epoch_019_discriminator.png', **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()

if __name__ == "__main__":
    main()
