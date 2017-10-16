from conditional_gan import make_generator, make_discriminator, CGAN
from gan.cmd import parser_with_default_args
from gan.train import Trainer

import numpy as np
from pose_dataset import PoseHMDataset
from pose_gan import POSE_GAN
from keras.models import load_model, Input
from gan.layer_utils import LayerNorm
from skimage.transform import resize
import pose_utils
from skimage.io import imsave
from scipy.ndimage.filters import gaussian_filter


def main():
    parser = parser_with_default_args()
    parser.add_argument("--pose_generator", default="output/checkpoints_bbox_10_G_S/epoch_2499_generator.h5", help="Generator of structure")
    parser.add_argument("--output_img", default='cgan_bbox_10.png', help='Name of the output image')
    args = parser.parse_args()
            
    generator = make_generator(Input((128, 64, 1)), Input((128, 64, 3)))
    if args.generator_checkpoint is not  None:
        generator.load_weights(args.generator_checkpoint)
    
    pose_generator = load_model(args.pose_generator)    
    
    dataset = PoseHMDataset(args.input_folder, None, args.batch_size, (64,), (128, 64))
    

    poses = pose_generator.predict([np.random.normal(size = (args.batch_size, 64))])
    
    def deprocess_array(X):
        X = X / 2 + 0.5
        X = X.reshape((X.shape[0], 10, 4))
        X[...,0::2] *= 128 - 0.1
        X[...,1::2] *= 64 - 0.1
        return X
    poses = deprocess_array(poses)
    
    import keras.backend as K
    import tensorflow as ktf
    img_size_init = (int(1.05 * 128), int(1.05 * 64))
    K._LEARNING_PHASE = ktf.constant(1)
    
    poses = np.array([pose_utils.bbox_to_map(pose, (128, 64)) for pose in poses])
    poses = poses[..., np.newaxis]
    poses = [resize(pose, img_size_init, preserve_range=True) for pose in poses]
    
    poses = np.array([dataset._random_crop(b, b)[0] for b in poses])
    batch = dataset.next_generator_sample()
    batch[0] = poses
    
    img = dataset.display(generator.predict(batch), batch, row=8, col=8)
    imsave(args.output_img, img[:,:2 * img.shape[1]/3])

    
if __name__ == "__main__":
    main()