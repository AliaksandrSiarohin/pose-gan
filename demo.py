from compute_coordinates import cordinates_from_image_file
from create_pairs_dataset import filter_not_valid
import cmd
import os
from shutil import copy, rmtree
import pandas as pd
from pose_dataset import PoseHMDataset
from conditional_gan import make_generator
from tqdm import tqdm
from test import generate_images, save_images
from keras.models import load_model
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np

if __name__ == "__main__":
    args = cmd.args()

    dataset = 'tmp'

    target_images_folder = 'data/target-images'
    source_image = 'data/source-image.jpg'
    bg_image = 'data/bg-image.jpg'

    #For target images use kp from previous frame if not detected
    interpolate = True

    args.images_dir_test = 'data/' + dataset + '-dataset/test'
    args.annotations_file_test = 'data/' + dataset + '-annotation-test.csv'
    args.pairs_file_test = 'data/' + dataset + '-pairs-test.csv'
    args.bg_images_dir_test = 'data/' + dataset + '-dataset/test-bg'

    args.images_dir_train = 'data/' + dataset + '-dataset/train'
    args.annotations_file_train = 'data/' + dataset + '-annotation-train.csv'
    args.pairs_file_train = 'data/' + dataset + '-pairs-train.csv'
    args.bg_images_dir_train = 'data/' + dataset + '-dataset/train-bg'

    f = open(args.annotations_file_train, 'w')
    print >>f, 'name:keypoints_y:keypoints_x'
    f.close()

    f = open(args.pairs_file_train, 'w')
    print >>f, 'from,to'
    f.close()

    print ("Annotate image keypoints...")
    if os.path.exists(args.images_dir_test):
        rmtree(args.images_dir_test)
    os.makedirs(args.images_dir_test)

    kp_model = load_model(args.pose_estimator)
    images_to_annotate = [os.path.join(target_images_folder, name) for name in os.listdir(target_images_folder)]
    images_to_annotate.sort()
    images_to_annotate.append(source_image)

    result_file = open(args.annotations_file_test, 'w')
    print >>result_file, 'name:keypoints_y:keypoints_x'
    prev_pose_cord = -np.ones((18, 2))
    for i, image_name in tqdm(enumerate(images_to_annotate)):
        img = imread(image_name)
        imsave(os.path.join(args.images_dir_test, os.path.basename(image_name)), resize(img, (128, 64)))
        pose_cords = cordinates_from_image_file(os.path.join(args.images_dir_test, os.path.basename(image_name)),
                                                    kp_model)

        if interpolate and image_name != source_image:
            pose_cords[pose_cords == -1] = prev_pose_cord[pose_cords == -1]
            prev_pose_cord = pose_cords


        print >> result_file, "%s: %s: %s" % (os.path.basename(image_name),
                                              str(list(pose_cords[:, 0])), str(list(pose_cords[:, 1])))
        result_file.flush()
    result_file.close()

    print ("Create pairs dataset...")
    df_keypoints = pd.read_csv(args.annotations_file_test, sep=':')
    df = filter_not_valid(df_keypoints)
    fr_list, to_list = [], []
    valid_names = set(df['name'])
    for img_to in os.listdir(target_images_folder):
        if img_to in valid_names:
            fr_list.append(os.path.basename(source_image))
            to_list.append(img_to)
    pair_df = pd.DataFrame(index=range(len(fr_list)))
    pair_df['from'] = fr_list
    pair_df['to'] = to_list

    print ('Number of pairs: %s' % len(pair_df))
    pair_df.to_csv(args.pairs_file_test, index=False)

    print ("Create bg images...")
    if os.path.exists(args.bg_images_dir_test):
        rmtree(args.bg_images_dir_test)
    os.makedirs(args.bg_images_dir_test)

    for img_to in os.listdir(target_images_folder):
        if img_to in valid_names:
            copy(bg_image, os.path.join(args.bg_images_dir_test, img_to.replace('.jpg', '_BG.jpg')))

    print ("Generating images...")
    dataset = PoseHMDataset(test_phase=True, **vars(args))
    generator = make_generator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type,
                               args.warp_agg, args.use_bg, args.pose_rep_type)
    assert (args.generator_checkpoint is not None)
    generator.load_weights(args.generator_checkpoint)

    print ("Generate images...")
    input_images, target_images, generated_images, names = generate_images(dataset, generator, args.use_input_pose)
    print ("Save images to %s..." % (args.generated_images_dir, ))
    save_images(input_images, target_images, generated_images, names, args.generated_images_dir)
