import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import pylab as plt

from skimage.transform import resize
from skimage import img_as_ubyte
from gan.dataset import FolderDataset

from structure import StuctureDataset
import pose_utils
import pandas as pd

class PoseHMDataset(FolderDataset):
    def __init__(self, image_dir, pose_generator, batch_size, noise_size, image_size):
        super(PoseHMDataset, self).__init__(image_dir, batch_size, noise_size, image_size)
        self._batches_before_shuffle = int(self._image_names.shape[0] // self._batch_size)
        self._pose_generator = pose_generator
        
    def _deprocess_pose_array(self, X):
        X = X / 2 + 0.5
        X = X.reshape((X.shape[0], 18, 2))
        X[...,0] *= self._image_size[0] - 0.1
        X[...,1] *= self._image_size[1] - 0.1
        return X
    
    def _load_pose_array(self, joints, for_disc=False):
        pose_list = [resize(pose_utils.cords_to_map(joint, self._image_size), 
                        (self._image_size[0] / 8, self._image_size[1] / 8), preserve_range=True) for joint in joints]
        return np.array(pose_list)
        
    def next_generator_sample(self):
        index = np.random.choice(self._image_names.shape[0], size = self._batch_size)
        joints = self._deprocess_pose_array(self._pose_generator.predict(np.random.normal(size = (self._batch_size,64) ))) 
        noise = super(PoseHMDataset, self).next_generator_sample()[0]
        pose_array = self._load_pose_array(joints)
        return [noise, pose_array]
    
    def _load_discriminator_data(self, index):
        images_batch_128_64 = super(PoseHMDataset, self)._load_discriminator_data(index)[0]
        images_batch_64_32 = np.array([resize(img, (img.shape[0] / 2, img.shape[1] / 2), preserve_range=True)  
                              for img in images_batch_128_64])
        images_batch_32_16 = np.array([resize(img, (img.shape[0] / 4, img.shape[1] / 4), preserve_range=True)  
                              for img in images_batch_128_64])
        return [images_batch_128_64, images_batch_64_32, images_batch_32_16]
        
    def display(self, output_batch, input_batch, row=8, col=1):
        pose_batch = input_batch[1]
        pose_images = np.array([pose_utils.draw_pose_from_map(resize(pose, self._image_size, order=1, preserve_range=True))[0]
                                      for pose in pose_batch])
        pose_masks = np.array([pose_utils.draw_pose_from_map(resize(pose, self._image_size, order=1, preserve_range=True))[1]
                                      for pose in pose_batch])
        result_images = []
        for one_res_batch in output_batch:
            resized_batch = np.array([resize(img, self._image_size, preserve_range=True) for img in one_res_batch])
            result_image = super(PoseHMDataset, self).display(resized_batch, row=row, col=col)
            result_images.append(result_image)
        
        pose_result_image = super(FolderDataset, self).display(pose_images, row=row, col=col)
        pose_masks = np.expand_dims(pose_masks, axis=3)
        pose_result_mask = super(FolderDataset, self).display(pose_masks, row=row, col=col)
        result_with_pose = result_images[0].copy()
        pose_result_mask = np.squeeze(pose_result_mask, axis=2)
        result_with_pose[pose_result_mask] = pose_result_image[pose_result_mask]
        
        return np.concatenate(np.array([result_with_pose] + result_images), axis=1)
    
    
# import pandas as pd
# import json
# class PoseDataset(UGANDataset):
#     def __init__(self, input_dir, batch_size, noise_size, image_size, pose_anotations):
#         super(PoseDataset, self).__init__(batch_size, noise_size)
#         self._image_size = image_size
#         self._pose_anotations_df = pd.read_csv(pose_anotations, sep = ':')
#         self._batches_before_shuffle = int(len(self._pose_anotations_df) // self._batch_size)        
#         self._input_dir = input_dir
        
#     def _extract_keypoints_array(self, sample_index):
#         keypoints_x = np.array([json.loads(keypoints) 
#                         for keypoints in self._pose_anotations_df.iloc[sample_index]['keypoints_x']])
#         keypoints_y = np.array([json.loads(keypoints) 
#                         for keypoints in self._pose_anotations_df.iloc[sample_index]['keypoints_y']])
#         keypoints_x = np.expand_dims(keypoints_x, 2)
#         keypoints_y = np.expand_dims(keypoints_y, 2)
#         keypoints = np.concatenate([keypoints_y, keypoints_x], axis = 2)
#         return keypoints
        
#     def next_generator_sample(self):
#         noise = super(PoseDataset, self).next_generator_sample()        
#         sample_index = np.random.choice(len(self._pose_anotations_df), size = self._batch_size)       
#         keypoints = self._extract_keypoints_array(sample_index)        
#         return noise, keypoints    
    
#     def _load_discriminator_data(self, index):
#         images_batch = np.array([resize(plt.imread(os.path.join(self._input_dir, img_name)), self._image_size) * 2 - 1
#                                  for img_name in self._pose_anotations_df.iloc[index]['name']])
#         keypoints = self._extract_keypoints_array(index)
#         return images_batch, keypoints
    
#     def _shuffle_discriminator_data(self):
#         self._pose_anotations_df = self._pose_anotations_df.sample(frac=1)

        
#     def display(self, batch, pose = None, row=8, col=8):       
#         height, width = batch.shape[1], batch.shape[2]
#         total_width, total_height = width * col, height * row
#         result_image = np.empty((total_height, total_width, batch.shape[3]))
#         batch_index = 0
#         for i in range(row):
#             for j in range(col):
#                 image = np.copy(batch[batch_index])
#                 if pose is not None:
#                     for joint_index in range(pose.shape[1]):
#                         joint_cord = pose[batch_index][joint_index]
#                         if joint_cord[0] >= height or joint_cord[1] >= width or joint_cord[0] < 0 or joint_cord[1] < 0:
#                             continue
#                         image[joint_cord[0], joint_cord[1]] = (1, 0, 0)
                            
#                 result_image[(i * height):((i+1)*height), (j * width):((j+1)*width)] = image
#                 batch_index += 1
#         result_image = img_as_ubyte((result_image + 1) / 2)
#         return result_image