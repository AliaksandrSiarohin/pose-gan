from conditional_gan import make_generator
import cmd
from pose_dataset import PoseHMDataset

import numpy as np

from tqdm import tqdm
from skimage.io import imsave
import os

import pandas as pd
import pose_transform
import pose_utils
from itertools import permutations
from shutil import copy
from collections import defaultdict

def filter_not_valid(df_keypoints):
    def check_valid(x):
        kp_array = pose_utils.load_pose_cords_from_strings(x['keypoints_y'], x['keypoints_x'])
        distractor = x['name'].startswith('-1') or x['name'].startswith('0000')
        return pose_transform.check_valid(kp_array) and not distractor
    return df_keypoints[df_keypoints.apply(check_valid, axis=1)].copy()


def make_pairs(df, pairs_for_each=10):
    fr, to = [], []
    for image_name in df['name']:
        fr_names = [image_name] * pairs_for_each
        to_names = df['name'].sample(n=pairs_for_each)
        fr += list(fr_names)
        to += list(to_names)
    pair_df = pd.DataFrame(index=range(len(fr)))
    pair_df['from'] = fr
    pair_df['to'] = to
    return pair_df


def generate_images(dataset, generator, use_input_pose, out_dir, store_train_images):
    number = 0

    def deprocess_image(img):
        return (255 * ((img + 1) / 2.0)).astype(np.uint8)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for _ in tqdm(range(dataset._pairs_file_test.shape[0])):
        number += 1
        batch, name = dataset.next_generator_sample_test(with_names=True)
        out = generator.predict(batch)
        #from_image = deprocess_image(batch[0])
        out_index = 2 if use_input_pose else 1
        #to_image = deprocess_image(batch[out_index])
        generated_image = deprocess_image(out[out_index])
        out = np.squeeze(generated_image)# np.concatenate([from_image, to_image, generated_image], axis=1))
        name = name.iloc[0]['from'].replace('.jpg', 'g' + str(number) + '.jpg')
        imsave(os.path.join(out_dir, name), out)

    if store_train_images:
       for name in tqdm(os.listdir(dataset._images_dir_train)):
           copy(os.path.join(dataset._images_dir_train, name), out_dir)

def create_train_file(generated_images_folder, train_file_name, generated_as_separate):
    train_f = open(train_file_name, 'w')
    cls_to_num = {}
    for name in os.listdir(generated_images_folder):
        cls = name.split('_')[0]
        cls = int(cls)
        if cls not in cls_to_num:        
            cls_to_num[cls] = len(cls_to_num)
#    print cls_to_num
    for name in os.listdir(generated_images_folder):
        attr = name.replace('.', '_').split('_')
        cls, index = attr[0], attr[-2]
        cls = int(cls)
        if generated_as_separate and ('g' in index):
            num = cls_to_num[cls] + len(cls_to_num)
        else:
            num = cls_to_num[cls]
        print >>train_f, "%s %s" % ('dataset/bounding_box_train/' + name, num)
   
    train_f.close()

def test():
    args = cmd.args()

    args.images_dir_test = args.images_dir_train
    args.pairs_file_test = 'data/market-re-id-pairs.csv'
    pairs_for_each = 2
    train_file_name = 'train.txt'
    store_train_images = True
    generated_as_separate = False
    
    df_keypoints = pd.read_csv(args.annotations_file_train, sep=':')
    df = filter_not_valid(df_keypoints)

    print ('Compute pair for train re-id...')
    pairs_df_train = make_pairs(df, pairs_for_each)
    print ('Number of pairs: %s' % len(pairs_df_train))
    pairs_df_train.to_csv('data/market-re-id-pairs.csv', index=False)
    

    dataset = PoseHMDataset(test_phase=True, **vars(args))
    generator = make_generator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type,
                               args.warp_agg, args.use_bg, args.pose_rep_type)
    assert (args.generator_checkpoint is not None)
    generator.load_weights(args.generator_checkpoint)

    print ("Generate images...")
    generate_images(dataset, generator, args.use_input_pose, args.generated_images_dir, store_train_images=store_train_images)
    
    print ("Creating train file...")
    create_train_file(args.generated_images_dir, train_file_name, generated_as_separate)

test()
