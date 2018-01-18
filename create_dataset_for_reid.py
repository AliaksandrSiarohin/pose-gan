from conditional_gan import make_generator
import cmd
from pose_dataset import PoseHMDataset

import numpy as np

from tqdm import tqdm
from skimage.io import imsave
import os


def generate_images(dataset, generator, use_input_pose, out_dir):
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


def test():
    args = cmd.args()

    args.images_dir_test = args.images_dir_train
    args.pairs_file_test = 'data/market-re-id-pairs.csv'

    dataset = PoseHMDataset(test_phase=True, **vars(args))
    generator = make_generator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type, args.warp_agg)
    assert (args.generator_checkpoint is not None)
    generator.load_weights(args.generator_checkpoint)

    print ("Generate images...")
    generate_images(dataset, generator, args.use_input_pose, args.generated_images_dir)


test()
