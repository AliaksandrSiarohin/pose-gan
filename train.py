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
    parser.add_argument("--pose_estimator", default="pose-anotations/pose_estimator.h5", help="InceptionV3 based pose estimator")
    parser.add_argument("--pose_penalty_weight", default=0.1, type=int, help="Weight of pose penalty")
    parser.add_argument("--pose_anotations", default='pose-anotations/pose_anotations.csv', help="Csv file with pose anotations")
    parser.add_argument("--input_folder", default='../market-dataset/bounding_box_train', help='Folder with real images for training')
    parser.add_argument("--display_ratio", default=1,  help='Number of epochs between ploting')
    args = parser.parse_args()
    return args

args = get_args()
assert K.image_data_format() == 'channels_last'

#from gan import GAN
#import mnist_architectures as architectures
#from dataset import MNISTDataset as Dataset

# from wgan_gp import WGAN_GP as GAN
# import small_res_architectures as architectures
# from dataset import FolderDataset as Dataset

from pose_gan import POSE_GAN as GAN
import pose_guided_architectures as architectures
from dataset import PoseDataset as Dataset
from pose_guided_architectures import LayerNorm, PoseMapFromCordinatesLayer

from tqdm import tqdm
from keras.models import load_model
import tensorflow as tf
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def save_image(image, output_directory, title):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.imsave(os.path.join(output_directory, title), image,  cmap=plt.cm.gray)
    
def save_model(model, output_directory, title):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    model.save(os.path.join(output_directory, title))
    
def train():   
    K.set_learning_phase(1)
    pose_estimator = load_model(args.pose_estimator)
    
    if args.generator_checkpoint is None:
        generator = architectures.make_generator()
    else:
        generator = load_model(args.generator_checkpoint, custom_objects = 
                                   {'PoseMapFromCordinatesLayer' : PoseMapFromCordinatesLayer,   'LayerNorm' : LayerNorm})    
    print ("Generator Summary:")
    generator.summary()
       
    if args.discriminator_checkpoint is None:
        discriminator = architectures.make_discriminator()
    else:
        discriminator = load_model(args.discriminator_checkpoint, custom_objects = 
                                   {'PoseMapFromCordinatesLayer' : PoseMapFromCordinatesLayer,   'LayerNorm' : LayerNorm})
    print ("Discriminator Summary:")
    discriminator.summary()
    
    noise_size = 64
    pose_size = (16, 2)
    image_size = (128, 64, 3)
    generator_model, discriminator_model = GAN(generator, discriminator, pose_estimator,
                                               Input(shape=(noise_size, )), Input(shape=pose_size, dtype='int32'),
                                               Input(shape=image_size), Input(shape=pose_size, dtype='int32'),
                                               cmd_args = args ).compile_models()
    
    dataset = Dataset(batch_size = args.batch_size, noise_size = noise_size, 
                      input_dir=args.input_folder, image_size=image_size[:2], pose_anotations=args.pose_anotations)
    
    gt_image, gt_pose = dataset.next_discriminator_sample()
    save_image(dataset.display(gt_image, gt_pose), args.output_dir, 'gt_data.png')
    
    noise_batch, pose_batch = dataset.next_generator_sample() 
    image = dataset.display(generator.predict_on_batch([noise_batch, pose_batch]), pose_batch)
    save_image(image, args.output_dir, 'epoch_{}.png'.format(0))
  
    for epoch in range(args.number_of_epochs):        
        print("Epoch: ", epoch)
        discriminator_loss_list = []
        generator_loss_list = []
        
        for i in tqdm(range(int(dataset._batches_before_shuffle // args.training_ratio))):
            for j in range(args.training_ratio):
                image_batch, pose_discriminator_batch = dataset.next_discriminator_sample()
                noise_batch, pose_generator_batch = dataset.next_generator_sample()
                #All zeros as ground truth because it`s not used
                loss = discriminator_model.train_on_batch([image_batch, pose_discriminator_batch, 
                                                           noise_batch, pose_generator_batch],
                                                          np.zeros([args.batch_size]))
                discriminator_loss_list.append(loss)
            
            noise_batch, pose_batch = dataset.next_generator_sample()
            loss = generator_model.train_on_batch([noise_batch, pose_batch], np.zeros([args.batch_size]))
            generator_loss_list.append(loss)
            
          
        print ("Discriminator loss: ", np.mean(np.array(discriminator_loss_list), axis = 0))
        print ("Generator loss: ", np.mean(np.array(generator_loss_list), axis = 0))
        
        if (epoch + 1) % args.display_ratio == 0:
            noise_batch, pose_batch = dataset.next_generator_sample() 
            image = dataset.display(generator.predict_on_batch([noise_batch, pose_batch]), pose_batch)
            save_image(image, args.output_dir, 'epoch_{}.png'.format(epoch))
        
        
        if (epoch + 1) % args.checkpoint_ratio == 0:
            save_model(generator, args.checkpoints_dir, 'epoch_{}_generator.h5'.format(epoch))
            save_model(discriminator, args.checkpoints_dir, 'epoch_{}_discriminator.h5'.format(epoch))
            
if __name__ == "__main__":
    train()
