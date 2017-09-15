import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import pylab as plt

from skimage.transform import resize
from skimage import img_as_ubyte

class UGANDataset(object):
    def __init__(self, batch_size, noise_size):
        self._batch_size = batch_size
        self._noise_size = noise_size
        self._batches_before_shuffle = 1000
        self._current_batch = 0
        
    def next_generator_sample(self):
        return [np.random.normal(size=(self._batch_size,) + self._noise_size)]
    
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

    def display(self, output_batch, input_batch = None, row=8, col=8):
        batch = output_batch
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
        return [self._X[index]]
    
    def _shuffle_discriminator_data(self):
        np.random.shuffle(self._X)
    
class FolderDataset(UGANDataset):
    def __init__(self, input_dir, batch_size, noise_size, image_size):
        super(FolderDataset, self).__init__(batch_size, noise_size)        
        self._image_names = np.array(os.listdir(input_dir))
        self._input_dir = input_dir
        self._image_size = image_size
        self._batches_before_shuffle = int(self._image_names.shape[0] // self._batch_size)        
        
    def _preprocess_image(self, img):
        return resize(img, self._image_size) * 2 - 1
    
    def _deprocess_image(self, img):
        return img_as_ubyte((img + 1) / 2)
        
    def _load_discriminator_data(self, index):
        return [np.array([self._preprocess_image(plt.imread(os.path.join(self._input_dir, img_name)))
                          for img_name in self._image_names[index]])]
    
    def _shuffle_discriminator_data(self):
        np.random.shuffle(self._image_names)
        
    def display(self, output_batch, input_batch = None, row=8, col=8):
        image = super(FolderDataset, self).display(output_batch, row, col)
        return self._deprocess_image(image)
    
    

