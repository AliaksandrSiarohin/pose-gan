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
    
    
class PoseHMDataset(UGANDataset):
    def __init__(self, image_dir, pose_dir, batch_size, noise_size):
        super(PoseHMDataset, self).__init__(batch_size, noise_size)
        self._image_names = np.array(os.listdir(image_dir))
        self._batches_before_shuffle = int(self._image_names.shape[0] // self._batch_size)
        self._pose_dir = pose_dir
        self._image_dir = image_dir
        
    def _load_pose_array(self, index):
        names = self._image_names[index]
        return np.array([np.load(os.path.join(self._pose_dir, name + '.npy')) for name in names], dtype='float32')[..., :18]   
        
    def next_generator_sample(self):
        index = np.random.choice(self._image_names.shape[0], size = self._batch_size)
        noise = np.random.normal(size=(self._batch_size, ) + self._noise_size)
        return noise, self._load_pose_array(index)
    
    def _load_discriminator_data(self, index):
        images = np.array([2 * (plt.imread(os.path.join(self._image_dir, img_name))/255.0 - 0.5)
                           for img_name in self._image_names[index]])
        poses = self._load_pose_array(index)
        return images, poses
    
    
    def _shuffle_discriminator_data(self):
        np.random.shuffle(self._image_names)
        
    def display(self, batch, pose = None, row=8, col=8):
        if len(pose.shape) == 4:
            new_pose = np.empty((pose.shape[0], pose.shape[-1], 2), dtype='int32')
            for i in range(pose.shape[0]):
                m = resize(pose[i], (128, 64), order=1, preserve_range = True)
                y, x, _ = np.where(m == m.max(axis = (0, 1)))
                new_pose[i,:,0] = y
                new_pose[i,:,1] = x
            pose = new_pose
        height, width = batch.shape[1], batch.shape[2]
        total_width, total_height = width * col, height * row
        result_image = np.empty((total_height, total_width, batch.shape[3]))
        batch_index = 0
        for i in range(row):
            for j in range(col):
                image = np.copy(batch[batch_index])
                if pose is not None:
                    for joint_index in range(pose.shape[1]):
                        joint_cord = pose[batch_index][joint_index]
                        if joint_cord[0] >= height or joint_cord[1] >= width or joint_cord[0] < 0 or joint_cord[1] < 0:
                            continue
                        image[joint_cord[0], joint_cord[1]] = (1, 0, 0)
                            
                result_image[(i * height):((i+1)*height), (j * width):((j+1)*width)] = image
                batch_index += 1
        result_image = img_as_ubyte((result_image + 1) / 2)
        return result_image
    
    
import pandas as pd
import json
class PoseDataset(UGANDataset):
    def __init__(self, input_dir, batch_size, noise_size, image_size, pose_anotations):
        super(PoseDataset, self).__init__(batch_size, noise_size)
        self._image_size = image_size
        self._pose_anotations_df = pd.read_csv(pose_anotations, sep = ':')
        self._batches_before_shuffle = int(len(self._pose_anotations_df) // self._batch_size)        
        self._input_dir = input_dir
        
    def _extract_keypoints_array(self, sample_index):
        keypoints_x = np.array([json.loads(keypoints) 
                        for keypoints in self._pose_anotations_df.iloc[sample_index]['keypoints_x']])
        keypoints_y = np.array([json.loads(keypoints) 
                        for keypoints in self._pose_anotations_df.iloc[sample_index]['keypoints_y']])
        keypoints_x = np.expand_dims(keypoints_x, 2)
        keypoints_y = np.expand_dims(keypoints_y, 2)
        keypoints = np.concatenate([keypoints_y, keypoints_x], axis = 2)
        return keypoints
        
    def next_generator_sample(self):
        noise = super(PoseDataset, self).next_generator_sample()        
        sample_index = np.random.choice(len(self._pose_anotations_df), size = self._batch_size)       
        keypoints = self._extract_keypoints_array(sample_index)        
        return noise, keypoints    
    
    def _load_discriminator_data(self, index):
        images_batch = np.array([resize(plt.imread(os.path.join(self._input_dir, img_name)), self._image_size) * 2 - 1
                                 for img_name in self._pose_anotations_df.iloc[index]['name']])
        keypoints = self._extract_keypoints_array(index)
        return images_batch, keypoints
    
    def _shuffle_discriminator_data(self):
        self._pose_anotations_df = self._pose_anotations_df.sample(frac=1)

        
    def display(self, batch, pose = None, row=8, col=8):       
        height, width = batch.shape[1], batch.shape[2]
        total_width, total_height = width * col, height * row
        result_image = np.empty((total_height, total_width, batch.shape[3]))
        batch_index = 0
        for i in range(row):
            for j in range(col):
                image = np.copy(batch[batch_index])
                if pose is not None:
                    for joint_index in range(pose.shape[1]):
                        joint_cord = pose[batch_index][joint_index]
                        if joint_cord[0] >= height or joint_cord[1] >= width or joint_cord[0] < 0 or joint_cord[1] < 0:
                            continue
                        image[joint_cord[0], joint_cord[1]] = (1, 0, 0)
                            
                result_image[(i * height):((i+1)*height), (j * width):((j+1)*width)] = image
                batch_index += 1
        result_image = img_as_ubyte((result_image + 1) / 2)
        return result_image
