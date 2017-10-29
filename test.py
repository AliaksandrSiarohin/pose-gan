import os

from conditional_gan import make_generator
import cmd
from pose_dataset import PoseHMDataset

from gan.test import generate_images
from gan.inception_score import get_inception_score

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import numpy as np


def l1_score(pairs_df, generated_images, images_folder):
    score_list = []
    for df_row, generated_image in zip(pairs_df.iterrows(), generated_images):
        reference_image = imread(os.path.join(images_folder, df_row[1]['to']))
        score = np.abs(reference_image - generated_image).mean()
        score_list.append(score / 255.0)
    return np.mean(score_list)

def ssim_score(pairs_df, generated_images, images_folder):
    ssim_score_list = []
    for df_row, generated_image in zip(pairs_df.iterrows(), generated_images):
        reference_image = imread(os.path.join(images_folder, df_row[1]['to']))
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


def test():
    args = cmd.args()

    generator = make_generator(args.image_size, args.use_input_pose, args.warp_skip)
    assert (args.generator_checkpoint is not None)
    generator.load_weights(args.generator_checkpoint)

    dataset = PoseHMDataset(args.images_dir_test, 1, args.image_size, args.pairs_file_test,
                            args.annotations_file_test, args.use_input_pose, args.warp_skip, shuffle=False)

    pairs_df = dataset._pairs_file
    print ("Generate images...")
    generated_images = generate_images(dataset, generator, pairs_df.shape[0], out_index=-2)

    print ("Save images to %s..." % (args.generated_images_dir, ))
    save_images(pairs_df, generated_images, args.images_dir_test,
                args.generated_images_dir, args.generated_images_save_format)

    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(pairs_df, generated_images, args.images_dir_test)

    print ("Compute l1 score...")
    norm_score = l1_score(pairs_df, generated_images, args.images_dir_test)

    print ("Inception score = %s, SSIM score = %s, l1 score = %s" % (inception_score, structured_score, norm_score))



if __name__ == "__main__":
    test()


