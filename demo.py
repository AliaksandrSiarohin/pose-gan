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

if __name__ == "__main__":
    args = cmd.args()

    dataset = 'tmp'

    target_images_folder = 'data/target-images'
    source_image = 'data/source-image.jpg'
    bg_image = 'data/bg-image.jpg'

    crop_frames = False

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
    images_to_annotate.append(source_image)

    result_file = open(args.annotations_file_test, 'w')
    print >>result_file, 'name:keypoints_y:keypoints_x'

    for i, image_name in tqdm(enumerate(images_to_annotate)):

        if not crop_frames or image_name == source_image:
            img = imread(image_name)
            imsave(os.path.join(args.images_dir_test, os.path.basename(image_name)), resize(img, (128, 64)))
            pose_cords = cordinates_from_image_file(os.path.join(args.images_dir_test, os.path.basename(image_name)),
                                                    kp_model)
        else:
            pose_cords = cordinates_from_image_file(image_name, kp_model)
            if i == 0:
                y, x = pose_cords[:, 0], pose_cords[:, 1]

                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()

                image = imread(image_name)
                x_len = (x_max - x_min)
                y_len = (y_max - y_min)

                y_min -= int(0.2 * y_len)
                y_max += int(0.2 * y_len)
                y_len = (y_max - y_min)

                target_x_len = y_len / 2
                adj_x_len = int((target_x_len - x_len)/2)

                x_min -= adj_x_len
                x_max += adj_x_len

                x_min = max(0, x_min)
                y_min = max(0, y_min)

                print x_min, x_max, y_min, y_max
            image = imread(image_name)
            image_crop = image[y_min:y_max, x_min:x_max]
            mult_x = 64.0 / image_crop.shape[1]
            mult_y = 128.0 / image_crop.shape[0]
            image_crop = resize(image_crop, (128, 64))

            imsave(os.path.join(args.images_dir_test, os.path.basename(image_name)), image_crop)

            pose_cords[:, 0] -= y_min
            pose_cords[:, 1] -= x_min

            print pose_cords[:, 1]
            pose_cords[:, 0] = (pose_cords[:, 0] * mult_y).astype('int64')
            pose_cords[:, 1] = (pose_cords[:, 1] * mult_x).astype('int64')
            print pose_cords[:, 1]

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
