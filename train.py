import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from functools import partial
import matplotlib
matplotlib.use('Agg')
import pylab as plt

def get_args():
    parser = argparse.ArgumentParser(description="Improved Wasserstein GAN implementation for Keras.")
    parser.add_argument("--output_dir", default='generated_samples',
                        help="Directory with generated sample images")
    parser.add_argument("--batch_size", default=64, type=int, help='Size of the batch')
    parser.add_argument("--training_ratio", default=5, type=int,
                        help="The training ratio is the number of discriminator updates per generator update." + 
                        "The paper uses 5")
    parser.add_argument("--gradient_penalty_weight", default=10, type=float, help='Weight of gradient penalty loss')
    parser.add_argument("--number_of_epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--checkpoints_dir", default="checkpoints", help="Folder with checkpoints")
    parser.add_argument("--checkpoint_ratio", default=10, type=int, help="Number of epochs between consecutive checkpoints")
    parser.add_argument("--generator_checkpoint", default=None, help="Previosly saved model of generator")
    parser.add_argument("--discriminator_checkpoint", default=None, help="Previosly saved model of discriminator")
    args = parser.parse_args()
    return args

args = get_args()
assert K.image_data_format() == 'channels_last'

from wgan_gp import WGAN_GP as GAN
import mnist_architectures
from dataset import MNISTDataset
from tqdm import tqdm
from keras.models import load_model

def save_image(image, output_directory, title):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.imsave(os.path.join(output_directory, title), image,  cmap=plt.cm.gray)
    
def save_model(model, output_directory, title):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    model.save(os.path.join(output_directory, title))
    
def train():
    if args.generator_checkpoint is None:
        generator = mnist_architectures.make_generator()
    else:
        generator = load_model(args.generator_checkpoint)
    
    if args.discriminator_checkpoint is None:
        discriminator = mnist_architectures.make_discriminator()
    else:
        discriminator = load_model(args.discriminator_checkpoint)
        
    generator_model, discriminator_model = GAN(generator, discriminator, 
                                               Input(shape=(100,)), Input(shape=(28, 28, 1)), cmd_args = args ).compile_models()
    
    dataset = MNISTDataset(args.batch_size)
    for epoch in range(args.number_of_epochs):        
        print("Epoch: ", epoch)
        minibatches_size = args.batch_size * args.training_ratio
        discriminator_loss_list = []
        generator_loss_list = []
        
        for i in tqdm(range(int(dataset._batches_before_shuffle // args.training_ratio))):
            for j in range(args.training_ratio):
                generated_data = generator.predict_on_batch(dataset.next_generator_sample()[0])
                image_batch, y_batch = dataset.next_discriminator_sample(generated_data)
                loss = discriminator_model.train_on_batch(image_batch, y_batch)
                discriminator_loss_list.append(loss)
            
            noise_batch, y_batch = dataset.next_generator_sample()
            loss = generator_model.train_on_batch(noise_batch, y_batch)
            generator_loss_list.append(loss)
            
        image = dataset.display(generator.predict_on_batch(dataset.next_generator_sample()[0]))
        save_image(image, args.output_dir, 'epoch_{}.png'.format(epoch))
        print ("Discriminator loss: ", np.mean(discriminator_loss_list))
        print ("Generator loss: ", np.mean(generator_loss_list))
        
        if (epoch + 1) % args.checkpoint_ratio == 0:
            save_model(generator, args.checkpoints_dir, 'epoch_{}_generator.h5'.format(epoch))
            save_model(discriminator, args.checkpoints_dir, 'epoch_{}_discriminator.h5'.format(epoch))
train()
