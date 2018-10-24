import numpy as np

import matplotlib
matplotlib.use('Agg')

from gan.dataset import UGANDataset
import pose_utils
import pose_transform

from skimage.io import imread
import pandas as pd
import os

class PoseHMDataset(UGANDataset):
    def __init__(self, test_phase=False, **kwargs):
        super(PoseHMDataset, self).__init__(kwargs['batch_size'], None)
        self._test_phase = test_phase

        self._batch_size = 1 if self._test_phase else kwargs['batch_size']
        self._image_size = kwargs['image_size']
        self._images_dir_train = kwargs['images_dir_train']
        self._images_dir_test = kwargs['images_dir_test']

        self._bg_images_dir_train = kwargs['bg_images_dir_train']
        self._bg_images_dir_test = kwargs['bg_images_dir_test']

        self._pairs_file_train = pd.read_csv(kwargs['pairs_file_train'])
        self._pairs_file_test = pd.read_csv(kwargs['pairs_file_test'])

        self._annotations_file_test = pd.read_csv(kwargs['annotations_file_test'], sep=':')
        self._annotations_file_train = pd.read_csv(kwargs['annotations_file_train'], sep=':')

        self._annotations_file = pd.concat([self._annotations_file_test, self._annotations_file_train],
                                           axis=0, ignore_index=True)

        self._annotations_file = self._annotations_file.set_index('name')

        self._use_input_pose = kwargs['use_input_pose']
        self._warp_skip = kwargs['warp_skip']
        self._disc_type = kwargs['disc_type']
        self._tmp_pose = kwargs['tmp_pose_dir']
        self._use_bg = kwargs['use_bg']
        self._pose_rep_type = kwargs['pose_rep_type']
        self._cache_pose_rep = kwargs['cache_pose_rep']

        self._test_data_index = 0

        if not os.path.exists(self._tmp_pose):
            os.makedirs(self._tmp_pose)

        print ("Number of images: %s" % len(self._annotations_file))
        print ("Number of pairs train: %s" % len(self._pairs_file_train))
        print ("Number of pairs test: %s" % len(self._pairs_file_test))

        self._batches_before_shuffle = int(self._pairs_file_train.shape[0] // self._batch_size)

    def number_of_batches_per_epoch(self):
        return 1000

    def number_of_batches_per_validation(self):
        return len(self._pairs_file_test) // self._batch_size

    def compute_pose_map_batch(self, pair_df, direction):
        assert direction in ['to', 'from']
        batch = np.empty([self._batch_size] + list(self._image_size) + [18 if self._pose_rep_type == 'hm' else 3])
        i = 0
        for _, p in pair_df.iterrows():
            row = self._annotations_file.loc[p[direction]]
            if self._cache_pose_rep:
                file_name = self._tmp_pose + p[direction] + self._pose_rep_type + '.npy'
                if os.path.exists(file_name):
                    pose = np.load(file_name)
                else:
                    kp_array = pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
                    if self._pose_rep_type == 'hm':
                        pose = pose_utils.cords_to_map(kp_array, self._image_size)
                    else:
                        pose = pose_transform.make_stickman(kp_array, self._image_size)
                    np.save(file_name, pose)
            else:
                    kp_array = pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
                    if self._pose_rep_type == 'hm':
                        pose = pose_utils.cords_to_map(kp_array, self._image_size)
                    else:
                        pose = pose_transform.make_stickman(kp_array, self._image_size)
            batch[i] = pose
            i += 1
        return batch

    def compute_cord_warp_batch(self, pair_df):
        if self._warp_skip == 'full':
            batch = [np.empty([self._batch_size] + [1, 8])]
        elif self._warp_skip == 'mask':
            batch = [np.empty([self._batch_size] + [10, 8]),
                     np.empty([self._batch_size, 10] + list(self._image_size))]
        else:
            batch = [np.empty([self._batch_size] + [72])]
        i = 0
        for _, p in pair_df.iterrows():
            fr = self._annotations_file.loc[p['from']]
            to = self._annotations_file.loc[p['to']]
            kp_array1 = pose_utils.load_pose_cords_from_strings(fr['keypoints_y'],
                                                                fr['keypoints_x'])
            kp_array2 = pose_utils.load_pose_cords_from_strings(to['keypoints_y'],
                                                                to['keypoints_x'])
            if self._warp_skip == 'mask':
                batch[0][i] = pose_transform.affine_transforms(kp_array1, kp_array2)
                batch[1][i] = pose_transform.pose_masks(kp_array2, self._image_size)
            elif self._warp_skip == 'full':
                batch[0][i] = pose_transform.estimate_uniform_transform(kp_array1, kp_array2)
            else: #sel._warp_skip == 'stn'
                batch[0][i][:36] = kp_array1.reshape((-1, ))
                batch[0][i][36:] = kp_array2.reshape((-1, ))
            i += 1
        return batch

    def _preprocess_image(self, image):
        return (image / 255 - 0.5) * 2

    def _deprocess_image(self, image):
        return (255 * (image + 1) / 2).astype('uint8')

    def load_image_batch(self, pair_df, direction='from'):
        assert direction in ['to', 'from']
        batch = np.empty([self._batch_size] + list(self._image_size) + [3])
        i = 0
        for _, p in pair_df.iterrows():
            if os.path.exists(os.path.join(self._images_dir_train, p[direction])):
                batch[i] = imread(os.path.join(self._images_dir_train, p[direction]))
            else:
                batch[i] = imread(os.path.join(self._images_dir_test, p[direction]))
            i += 1
        return self._preprocess_image(batch)

    def load_bg(self, pair_df):
        batch = np.empty([self._batch_size] + list(self._image_size) + [3])
        i = 0
        for _, p in pair_df.iterrows():
            name = p['to'].replace('.jpg', '_BG.jpg') 
            #print os.path.join(self._images_dir_train, name)
            if os.path.exists(os.path.join(self._bg_images_dir_train, name)):
                batch[i] = imread(os.path.join(self._bg_images_dir_train, name))
            else:
                batch[i] = imread(os.path.join(self._bg_images_dir_test, name))
            i += 1
        return self._preprocess_image(batch)

    def load_batch(self, index, for_discriminator, validation=False):
        if validation:
            pair_df = self._pairs_file_test.iloc[index]
        else:
            pair_df = self._pairs_file_train.iloc[index]
        result = [self.load_image_batch(pair_df, 'from')]
        if self._use_input_pose:
            result.append(self.compute_pose_map_batch(pair_df, 'from'))
        result.append(self.load_image_batch(pair_df, 'to'))
        result.append(self.compute_pose_map_batch(pair_df, 'to'))

        if self._use_bg:
            result.append(self.load_bg(pair_df))

        if self._warp_skip != 'none' and (not for_discriminator or self._disc_type == 'warp'):
            result += self.compute_cord_warp_batch(pair_df)

        return result

    def next_generator_sample(self):
        index = self._next_data_index()
        return self.load_batch(index, False)

    def next_generator_sample_test(self, with_names=False):
        index = np.arange(self._test_data_index, self._test_data_index + self._batch_size)
        index = index % self._pairs_file_test.shape[0]
        batch = self.load_batch(index, False, True)
        names = self._pairs_file_test.iloc[index]
        self._test_data_index += self._batch_size
        if with_names:
            return batch, names
        else:
            return batch

    def next_discriminator_sample(self):
        index = self._next_data_index()
        return self.load_batch(index, True)

    def _shuffle_data(self):
        self._pairs_file_train = self._pairs_file_train.sample(frac=1)
        
    def display(self, output_batch, input_batch):
        row = self._batch_size
        col = 1

        tg_app = self._deprocess_image(input_batch[0])
        tg_pose = input_batch[3 if self._use_input_pose else 2]
        tg_img = input_batch[2 if self._use_input_pose else 1]
        tg_img = self._deprocess_image(tg_img)

        res_img = self._deprocess_image(output_batch[2 if self._use_input_pose else 1])

        tg_app = super(PoseHMDataset, self).display(tg_app, None, row=row, col=col)

        if self._pose_rep_type == 'hm':
            pose_images = np.array([pose_utils.draw_pose_from_map(pose)[0] for pose in tg_pose])
        else:
            pose_images =  (255 * tg_pose).astype('uint8')
        tg_pose = super(PoseHMDataset, self).display(pose_images, None, row=row, col=col)

        tg_img = super(PoseHMDataset, self).display(tg_img, None, row=row, col=col)
        res_img = super(PoseHMDataset, self).display(res_img, None, row=row, col=col)

        return np.concatenate(np.array([tg_app, tg_pose, tg_img, res_img]), axis=1)
