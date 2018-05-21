from skimage.io import imread, imsave
import pose_utils
from cmd import args
import os
import re
import pandas as pd
import pose_transform
import numpy as np

args = args()
in_folder = 'ref_nn_fasion'
out_folder = 'ref_nn_fasion_separated'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

args.annotations_file_test =  'data/fasion-annotation-train.csv'
args.images_dir_test = 'data/fasion-dataset/train'

for n, img_pair in enumerate(os.listdir(in_folder)):
    m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)', img_pair)
    fr = m.groups()[0]
    to = m.groups()[1]

    gen_img = imread(os.path.join(in_folder, img_pair))
    gen_img = gen_img[:, (2 * args.image_size[1]):]

    df = pd.read_csv(args.annotations_file_test, sep=':')
    ano_fr = df[df['name'] == fr].iloc[0]
    ano_to = df[df['name'] == to].iloc[0]
    kp_fr = pose_utils.load_pose_cords_from_strings(ano_fr['keypoints_y'], ano_fr['keypoints_x'])
    kp_to = pose_utils.load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

    mask = pose_transform.pose_masks(kp_to, img_size=args.image_size).astype(bool)

    mask = np.array(reduce(np.logical_or, list(mask)))
    mask = mask.astype('float')

    pose_fr, _ = pose_utils.draw_pose_from_cords(kp_fr, args.image_size)
    pose_to, _ = pose_utils.draw_pose_from_cords(kp_to, args.image_size)

    cur_folder = os.path.join(out_folder, str(n))
    if not os.path.exists(cur_folder):
        os.makedirs(cur_folder)

    imsave(os.path.join(cur_folder, 'from.jpg'), imread(os.path.join(args.images_dir_test, fr)))
    imsave(os.path.join(cur_folder, 'to.jpg'), imread(os.path.join(args.images_dir_test, to)))

    imsave(os.path.join(cur_folder, 'frpose.jpg'), pose_fr)
    imsave(os.path.join(cur_folder, 'topose.jpg'), pose_to)

    # imsave(os.path.join(cur_folder, 'mask.jpg'), mask)

    imsave(os.path.join(cur_folder, 'gen.jpg'), gen_img)

    hm_from = pose_utils.cords_to_map(kp_fr, args.image_size).sum(axis=-1)
    hm_to = pose_utils.cords_to_map(kp_to, args.image_size).sum(axis=-1)

    # hm_from /= hm_from.max()
    # hm_to /= hm_to.max()
    #
    # imsave(os.path.join(cur_folder, 'hm_from.jpg'), hm_from)
    #
    # imsave(os.path.join(cur_folder, 'hm_to.jpg'), hm_to)


