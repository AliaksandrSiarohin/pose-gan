import os

from conditional_gan import make_generator
import cmd
from pose_dataset import PoseHMDataset

from gan.test import generate_images
from gan.inception_score import get_inception_score

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import numpy as np
import pandas as pd


def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def save_images(pairs_df, generated_images, images_folder, output_folder, img_save_format):
    assert ((set('iog') or set(img_save_format)) == set('iog'))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for df_row, generated_image in zip(pairs_df.iterrows(), generated_images):
        images = []
        res_name = '_'.join([df_row[1]['from'], df_row[1]['to']]) + '.jpg'
        for type in img_save_format:
            if type == 'i':
                img = imread(os.path.join(images_folder, df_row[1]['from']))
            elif type == 'o':
                img = imread(os.path.join(images_folder, df_row[1]['to']))
            else:
                img = generated_image
            images.append(img)
        imsave(os.path.join(output_folder, res_name), np.concatenate(images, axis=1))


def create_masked_image(pairs_df, images, annotation_file):
    import pose_utils
    masked_images = []
    df = pd.read_csv(annotation_file, sep=':')
    for df_row, image in zip(pairs_df.iterrows(), images):
        to = df_row[1]['to']
        ano_to = df[df['name'] == to].iloc[0]

        kp_to = pose_utils.load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = pose_utils.produce_ma_mask(kp_to, image.shape[:2])
        masked_images.append(image * mask[..., np.newaxis])

    return masked_images

def load_reference_images(pairs_df, images_folder):
    reference_images = []
    for df_row in pairs_df.iterrows():
        reference_images.append(imread(os.path.join(images_folder, df_row[1]['to'])))
    return reference_images

def test():
    args = cmd.args()

    generator = make_generator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type, args.warp_agg)
    assert (args.generator_checkpoint is not None)
    generator.load_weights(args.generator_checkpoint)

    dataset = PoseHMDataset(args.images_dir_test, 1, args.image_size, args.pairs_file_test,
                            args.annotations_file_test, args.use_input_pose, args.warp_skip, args.disc_type, args.tmp_pose_dir,
                            use_validation=0, shuffle=False)

    pairs_df = dataset._pairs_file
    print ("Generate images...")
    generated_images = generate_images(dataset, generator, pairs_df.shape[0], out_index=-2)
    reference_images = load_reference_images(pairs_df, args.images_dir_test)

    print ("Save images to %s..." % (args.generated_images_dir, ))
    save_images(pairs_df, generated_images, args.images_dir_test,
                args.generated_images_dir, args.generated_images_save_format)

    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s" % inception_score[0])

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, reference_images)
    print ("SSIM score %s" % structured_score)

    print ("Compute l1 score...")
    norm_score = l1_score(generated_images, reference_images)
    print ("L1 score %s" % norm_score)

    print ("Compute masked inception score...")
    generated_images_masked = create_masked_image(pairs_df, generated_images, args.annotations_file_test)
    reference_images_masked = create_masked_image(pairs_df, reference_images, args.annotations_file_test)
    inception_score_masked = get_inception_score(generated_images_masked)

    print ("Inception score masked %s" % inception_score_masked[0])
    print ("Compute masked SSIM...")
    structured_score_masked = ssim_score(generated_images_masked, reference_images_masked)
    print ("SSIM score masked %s" % structured_score_masked)

    print ("Inception score = %s, masked = %s; SSIM score = %s, masked = %s; l1 score = %s" %
           (inception_score, inception_score_masked, structured_score, structured_score_masked, norm_score))



if __name__ == "__main__":
    test()


