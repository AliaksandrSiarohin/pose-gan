import numpy as np

class UGANDataset(object):
    def __init__(self, batch_size, noise_size):
        self._generator_y = np.ones((batch_size, 1), dtype=np.float32)
        self._discriminator_y = np.concatenate([np.ones((batch_size, 1), dtype=np.float32),
                                          np.zeros((batch_size, 1), dtype=np.float32)])
        self._batch_size = batch_size
        self._noise_size = noise_size
        self._batches_before_shuffle = 1000
        self._current_batch = 0
        
    def next_generator_sample(self):
        return np.random.rand(self._batch_size, self._noise_size)
    
    def _load_discriminator_data(self, index):
        None
    
    def _shuffle_discriminator_data(self):
        None

    def next_discriminator_sample(self):
        self._current_batch %= self._batches_before_shuffle
        if self._current_batch == 0:
            self._shuffle_discriminator_data()
        index = np.arange(self._current_batch * self._batch_size, (self._current_batch + 1) * self._batch_size)
        self._current_batch += 1
        image_batch = self._load_discriminator_data(index)
        return image_batch
        

    def display(self, batch, row=8, col=8):
        height, width = batch.shape[1], batch.shape[2]
        total_width, total_height = width * col, height * row
        result_image = np.empty((total_height, total_width, batch.shape[3]))
        batch_index = 0
        for i in range(row):
            for j in range(col):
                result_image[(i * height):((i+1)*height), (j * width):((j+1)*width)] = batch[batch_index]
                batch_index += 1
        return result_image

    
class ArrayDataset(UGANDataset):
    def __init__(self, X, batch_size, noise_size):
        super(ArrayDataset, self).__init__(batch_size, noise_size)
        self._X = X
        self._batches_before_shuffle = int(X.shape[0] // self._batch_size)
    
    def _load_discriminator_data(self, index):
        return self._X[index]
    
    def _shuffle_discriminator_data(self):
        np.random.shuffle(self._X)
    
class MNISTDataset(ArrayDataset):
    def __init__(self, batch_size, noise_size = 100):
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate((X_train, X_test), axis=0)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        X = (X.astype(np.float32) - 127.5) / 127.5
        super(MNISTDataset, self).__init__(X, batch_size, noise_size)
        
    def display(self, batch, row=8, col=8):
        image = super(MNISTDataset, self).display(batch, row, col)
        image = (image * 127.5) + 127.5
        image = np.squeeze(np.round(image).astype(np.uint8))
        return image
    
import os
import pylab as plt
from skimage.transform import resize
from skimage import img_as_ubyte
class FolderDataset(UGANDataset):
    def __init__(self, input_dir, batch_size, noise_size, image_size):
        super(FolderDataset, self).__init__(batch_size, noise_size)        
        self._image_names = np.array([os.path.join(input_dir, name) for name in os.listdir(input_dir)])
        self._image_size = image_size
        self._batches_before_shuffle = int(self._image_names.shape[0] // self._batch_size)
        
    def next_generator_sample(self):
        return np.random.normal(size=(self._batch_size, self._noise_size))
    
    def _load_discriminator_data(self, index):
        return np.array([resize(plt.imread(img_name), self._image_size) * 2 - 1
                         for img_name in self._image_names[index]])
    
    def _shuffle_discriminator_data(self):
        np.random.shuffle(self._image_names)
        
    def display(self, batch, row=8, col=8):
        image = super(FolderDataset, self).display(batch, row, col)
        image = img_as_ubyte((image + 1) / 2)
        return image
    
import pandas as pd
import json
class PoseDataset(FolderDataset):
    def __init__(self, input_dir, batch_size, noise_size, image_size, pose_anotations):
        super(PoseDataset, self).__init__(input_dir, batch_size, noise_size, image_size)
        self._pose_anotations_df = pd.read_csv(pose_anotations, sep = ':')
    def next_generator_sample(self):
        noise = super(PoseDataset, self).next_generator_sample()
        
        sample_index = np.random.choice(len(self._pose_anotations_df), size = 64)        
        keypoints_x = np.array([json.loads(keypoints) 
                        for keypoints in self._pose_anotations_df.iloc[sample_index]['keypoints_x']])
        keypoints_y = np.array([json.loads(keypoints) 
                        for keypoints in self._pose_anotations_df.iloc[sample_index]['keypoints_y']])
        keypoints_x = np.expand_dims(keypoints_x, 2)
        keypoints_y = np.expand_dims(keypoints_y, 2)

        keypoints = np.concatenate([keypoints_y, keypoints_x], axis = 2)
        
        return noise, keypoints